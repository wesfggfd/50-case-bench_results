#!/usr/bin/env python3
import argparse
import json
import os
import stat
import textwrap
from pathlib import Path

import paramiko

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_BUNDLE_DIR = BASE_DIR / "official_bundle" / "public_standard_bundle"
DEFAULT_REMOTE_ROOT = "/root/SVG Generation"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("REMOTE_HOST", "u3ssh.casdao.com"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("REMOTE_PORT", "53413")))
    parser.add_argument("--user", default=os.environ.get("REMOTE_USER", "root"))
    parser.add_argument("--password", default=os.environ.get("REMOTE_PASSWORD"))
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--remote-root", default=os.environ.get("REMOTE_SVG_ROOT", DEFAULT_REMOTE_ROOT))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--upload-cache", action="store_true")
    return parser.parse_args()


def connect(args):
    if not args.password:
        raise RuntimeError("Missing remote password. Pass --password or set REMOTE_PASSWORD.")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=args.host, port=args.port, username=args.user, password=args.password, timeout=20)
    return client


def ensure_local_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def read_remote_manifest(client, remote_root: str):
    remote_script_path = "/tmp/export_public_text_manifest.py"
    script = textwrap.dedent(
        f"""
        import json
        from pathlib import Path
        from datasets import Video, load_dataset

        bench_files = {{
            "real": "{remote_root}/downloads/bench/MMLottieBench/data/real-00000-of-00001.parquet",
            "synthetic": "{remote_root}/downloads/bench/MMLottieBench/data/synthetic-00000-of-00001.parquet",
        }}
        result_root = Path("{remote_root}/results/official_rerun")
        dataset = load_dataset("parquet", data_files=bench_files)
        task_map = {{
            "Text-to-Lottie": "text2lottie",
            "Text-Image-to-Lottie": "text_image2lottie",
        }}
        items = []
        for split in ["real", "synthetic"]:
            rows = dataset[split].cast_column("video", Video(decode=False))
            for row in rows:
                task_type = row["task_type"]
                if task_type not in task_map:
                    continue
                task_key = task_map[task_type]
                json_path = result_root / f"{{split}}_{{task_key}}" / f"mmlottie_bench_{{split}}" / f"{{row['id']}}.json"
                if json_path.exists():
                    items.append({{
                        "id": row["id"],
                        "split": split,
                        "task_key": task_key,
                        "task_type": task_type,
                        "text": row["text"],
                    }})
        print(json.dumps(items, ensure_ascii=False))
        """
    ).strip() + "\n"
    sftp = client.open_sftp()
    with sftp.file(remote_script_path, "w") as f:
        f.write(script)
    sftp.close()
    cmd = (
        "bash -lc "
        + repr(
            "source /root/miniconda3/etc/profile.d/conda.sh && "
            "conda activate Qwen-svg-generation && "
            f"python {remote_script_path}"
        )
    )
    stdin, stdout, stderr = client.exec_command(cmd)
    out = stdout.read().decode("utf-8", "ignore")
    err = stderr.read().decode("utf-8", "ignore")
    client.exec_command(f"rm -f {remote_script_path}")
    if err:
        raise RuntimeError(err)
    return json.loads(out)


def download_file(sftp, remote_path: str, local_path: Path):
    ensure_local_dir(local_path.parent)
    sftp.get(remote_path, str(local_path))


def upload_tree(sftp, local_root: Path, remote_root: str):
    for dirpath, dirnames, filenames in os.walk(local_root):
        rel = Path(dirpath).relative_to(local_root)
        remote_dir = Path(remote_root) / rel
        try:
            sftp.stat(str(remote_dir))
        except FileNotFoundError:
            parts = remote_dir.parts
            current = Path(parts[0])
            for part in parts[1:]:
                current = current / part
                try:
                    sftp.stat(str(current))
                except FileNotFoundError:
                    sftp.mkdir(str(current))
        for filename in filenames:
            local_path = Path(dirpath) / filename
            remote_path = remote_dir / filename
            sftp.put(str(local_path), str(remote_path))


def download_bundle(args):
    bundle_dir = args.bundle_dir
    manifest_path = bundle_dir / "judge_manifest_public_standard.json"
    results_root = bundle_dir / "official_rerun_text"
    ensure_local_dir(results_root)
    client = connect(args)
    try:
        items = read_remote_manifest(client, args.remote_root)
        manifest_path.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        sftp = client.open_sftp()
        remote_tar = "/tmp/public_standard_official_rerun_text.tar.gz"
        tar_cmd = (
            "bash -lc "
            + repr(
                "cd "
                + f"'{args.remote_root}/results/official_rerun' "
                + "&& tar -czf /tmp/public_standard_official_rerun_text.tar.gz "
                + "real_text2lottie real_text_image2lottie synthetic_text2lottie synthetic_text_image2lottie"
            )
        )
        stdin, stdout, stderr = client.exec_command(tar_cmd)
        stdout.read()
        err = stderr.read().decode("utf-8", "ignore")
        if err:
            raise RuntimeError(err)
        local_tar = bundle_dir / "official_rerun_text.tar.gz"
        download_file(sftp, remote_tar, local_tar)
        sftp.close()
        import tarfile

        with tarfile.open(local_tar, "r:gz") as tar:
            tar.extractall(results_root)
        local_tar.unlink()
        client.exec_command(f"rm -f {remote_tar}")
        print(f"WROTE {manifest_path}")
        print(f"DOWNLOADED {len(items)} text-task result json files via tarball")
    finally:
        client.close()


def upload_cache(args):
    bundle_dir = args.bundle_dir
    cache_root = BASE_DIR / "judge_cache_public_standard_claude35_aihhhl"
    report_path = BASE_DIR / "judge_metrics_report_public_local_claude35_aihhhl.json"
    if not cache_root.exists():
        raise RuntimeError(f"Local cache root not found: {cache_root}")
    client = connect(args)
    try:
        sftp = client.open_sftp()
        remote_cache_root = f"{args.remote_root}/OmniLottie/reproduction_results/judge_cache_public_standard_claude35_aihhhl"
        remote_report_path = f"{args.remote_root}/OmniLottie/reproduction_results/judge_metrics_report_public_local_claude35_aihhhl.json"
        upload_tree(sftp, cache_root, remote_cache_root)
        if report_path.exists():
            sftp.put(str(report_path), remote_report_path)
        sftp.close()
        print(f"UPLOADED {cache_root} -> {remote_cache_root}")
        if report_path.exists():
            print(f"UPLOADED {report_path} -> {remote_report_path}")
    finally:
        client.close()


def main():
    args = parse_args()
    if not args.download and not args.upload_cache:
        args.download = True
    if args.download:
        download_bundle(args)
    if args.upload_cache:
        upload_cache(args)


if __name__ == "__main__":
    main()
