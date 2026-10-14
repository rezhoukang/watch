"""Download pinned official sources and verify/download the C-RADIOv4 weights."""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rtdetrv4
from project_paths import MODELS_DIR


def setup() -> None:
    repo = rtdetrv4.repo_dir()
    if not (repo / ".git").is_dir():
        if repo.exists() and any(repo.iterdir()):
            raise RuntimeError(f"源码目录非空，请设置 RTDETRV4_REPO 指向空目录：{repo}")
        repo.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "remote", "add", "origin",
                        "https://github.com/RT-DETRs/RT-DETRv4.git"], check=True)
        subprocess.run(["git", "-C", str(repo), "fetch", "--depth", "1", "origin",
                        rtdetrv4.REPO_REVISION], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", "--detach", "FETCH_HEAD"], check=True)
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if revision != rtdetrv4.REPO_REVISION:
        raise RuntimeError(f"源码版本为 {revision}，需要 {rtdetrv4.REPO_REVISION}；请设置 RTDETRV4_REPO 指向空目录重新安装")
    print(f"Official source: {repo} @ {revision}")

    weight = MODELS_DIR / rtdetrv4.WEIGHTS_NAME
    weight.parent.mkdir(parents=True, exist_ok=True)
    if weight.exists():
        with weight.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != rtdetrv4.WEIGHTS_SHA256:
            raise RuntimeError(f"已有权重 SHA256 不匹配，请检查文件（不会覆盖）：{weight}")
    else:
        url = ("https://huggingface.co/Awiros/person_and_head_detection/resolve/"
               f"{rtdetrv4.WEIGHTS_REVISION}/ckpts/rt-detrv4s-cradiov4-so400m/best_stg2.safetensors")
        partial = weight.with_suffix(".safetensors.part")
        try:
            print(f"Downloading {weight.name} ...")
            with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            with partial.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != rtdetrv4.WEIGHTS_SHA256:
                raise RuntimeError("下载权重的 SHA256 不匹配")
            partial.replace(weight)
        finally:
            partial.unlink(missing_ok=True)
    print(f"Verified checkpoint: {weight}")
    print("Install dependencies: python -m pip install -r requirements-rtdetrv4.txt")


if __name__ == "__main__":
    setup()
