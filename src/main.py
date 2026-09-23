from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUTPUT = ROOT / "output"


def main():
    print(f"Project root: {ROOT}")


if __name__ == "__main__":
    main()
