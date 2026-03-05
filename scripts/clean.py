from pathlib import Path
import os

TARGETS = [
    Path("data/features"),
    Path("data/processed"),
    Path("artifacts"),
]


def remove_directory(path: Path):
    if not path.exists():
        print(f"[CLEAN] Not found: {path}")
        return

    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(Path(root) / name)
        for name in dirs:
            os.rmdir(Path(root) / name)

    os.rmdir(path)
    print(f"[CLEAN] Removed: {path}")


def main():
    print("[CLEAN] Starting cleanup...")

    for target in TARGETS:
        remove_directory(target)

    print("[CLEAN] Cleanup finished.")


if __name__ == "__main__":
    main()
