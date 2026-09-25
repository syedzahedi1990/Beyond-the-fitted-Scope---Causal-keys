import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(root=ROOT):
    root = Path(root).resolve()
    manifest = json.loads((root / "RELEASE.json").read_text())
    checked = 0
    total = 0
    for name, expected in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe release path: " + name)
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError("Missing file or symlink: " + name)
        if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError("Release integrity mismatch: " + name)
        checked += 1
        total += expected["bytes"]
    return {"status": "PASS", "files_checked": checked, "bytes_checked": total,
            "scope": "Published code, configuration, datasets and prediction evidence"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))


if __name__ == "__main__":
    main()
