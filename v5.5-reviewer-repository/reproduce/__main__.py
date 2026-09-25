import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from reproduce.integrity import ROOT, verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("reproduced"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    verified = verify()
    if args.verify_only:
        print(json.dumps(verified, indent=2))
        return
    destination = args.output.resolve()
    if destination.exists():
        raise ValueError("Use a new output directory; existing results will not be overwritten")
    destination.mkdir(parents=True)
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    environment["MPLCONFIGDIR"] = str(destination / ".matplotlib")
    start = time.monotonic()
    commands = [["-m", "reproduce.behavior", "--output", str(destination / "behavior")],
                ["-m", "reproduce.mechanism", "--output", str(destination / "mechanism")],
                ["-m", "reproduce.figures", "--mechanism-dir", str(destination / "mechanism"), "--output", str(destination / "figures")]]
    try:
        for command in commands:
            print("Running " + " ".join(command[:2]), flush=True)
            subprocess.run([sys.executable, *command], cwd=ROOT, env=environment, check=True)
        receipt = {"status": "COMPLETE", "integrity": verified, "python": platform.python_version(),
                   "seconds": time.monotonic() - start, "model_execution": False,
                   "scope": "Saved per-example predictions to paper statistics and figures"}
        (destination / "COMPLETE.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2))
    except BaseException as error:
        (destination / "FAILED.json").write_text(json.dumps({"status": "FAILED", "exception": type(error).__name__, "message": str(error)}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
