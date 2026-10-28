"""Download research data, pin revision and record hashes. No synthetic fallback."""

import concurrent.futures
import hashlib
import json
import pathlib
import urllib.request
import zipfile
import os

root = (
    pathlib.Path(os.environ.get("SIGNALFEED_HOME", "/data1/zhuzy/signalfeed")) / "raw"
)
root.mkdir(parents=True, exist_ok=True)
revision = "2115068f457b577f93a268fffe22373e45727d89"


def download(split):
    url = f"https://huggingface.co/datasets/Recommenders/MIND/resolve/{revision}/MINDsmall_{split}.zip"
    dest = root / f"MINDsmall_{split}.zip"
    if not dest.exists():
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
    with zipfile.ZipFile(dest) as archive:
        for name in ("news.tsv", "behaviors.tsv"):
            archive.extract(name, root / split)
    return dict(
        split=split,
        url=url,
        sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
        bytes=dest.stat().st_size,
    )


if __name__ == "__main__":
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/msnews/MIND/master/MSR%20License_Data.pdf",
        root / "MSR_License_Data.pdf",
    )
    results = list(
        concurrent.futures.ThreadPoolExecutor(2).map(download, ["train", "dev"])
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "source": "Recommenders/MIND mirror; original Azure returned HTTP 409",
                "files": results,
            },
            indent=2,
        )
    )
    print(results)
