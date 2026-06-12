"""Ekzekuton te dy skriptet kryesore te projektit."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
SCRIPTS = ["classifiers_evaluation.py", "clustering_analysis.py"]


def main() -> None:
    for script in SCRIPTS:
        print("\n" + "=" * 60)
        print(f"Ekzekutimi: {script}")
        print("=" * 60)
        result = subprocess.run([sys.executable, str(ROOT / script)], check=False)
        if result.returncode != 0:
            print(f"Gabim gjate ekzekutimit te {script}")
            sys.exit(result.returncode)

    print("\nTe gjitha skriptet perfunduan me sukses.")
    print(f"Rezultatet: {ROOT / 'results'}")


if __name__ == "__main__":
    main()
