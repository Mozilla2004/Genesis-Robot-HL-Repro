"""Clean runs directory for Genesis Robot HL Repro v0.4.5."""

import os
import glob
import argparse


def clean_runs():
    """Clean runs directory to remove old results."""
    print("🧹 Cleaning runs directory...")

    # Files to remove
    files_to_remove = [
        "runs/trials.jsonl",
        "runs/summary.csv",
        "runs/sweep_results.csv",
        "runs/comparison_results.csv"
    ]

    # PNG files to remove
    png_files = glob.glob("runs/*.png")

    removed_count = 0

    # Remove specific files
    for file_pattern in files_to_remove:
        if os.path.exists(file_pattern):
            try:
                os.remove(file_pattern)
                print(f"  ✓ Removed {file_pattern}")
                removed_count += 1
            except Exception as e:
                print(f"  ✗ Failed to remove {file_pattern}: {e}")
        else:
            print(f"  - Skipped {file_pattern} (not found)")

    # Remove PNG files
    for png_file in png_files:
        try:
            os.remove(png_file)
            print(f"  ✓ Removed {png_file}")
            removed_count += 1
        except Exception as e:
            print(f"  ✗ Failed to remove {png_file}: {e}")

    # Ensure runs/.gitkeep exists
    gitkeep_path = "runs/.gitkeep"
    if not os.path.exists(gitkeep_path):
        try:
            os.makedirs("runs", exist_ok=True)
            with open(gitkeep_path, 'w') as f:
                f.write("# Keep runs directory in git\n")
            print(f"  ✓ Created {gitkeep_path}")
        except Exception as e:
            print(f"  ✗ Failed to create {gitkeep_path}: {e}")

    if removed_count == 0:
        print("🎯 Runs directory already clean")
    else:
        print(f"✅ Cleaned {removed_count} file(s)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Clean runs directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be cleaned without actually cleaning")

    args = parser.parse_args()

    if args.dry_run:
        print("🔍 Dry run mode - showing what would be cleaned:")
        files_to_check = [
            "runs/trials.jsonl",
            "runs/summary.csv",
            "runs/sweep_results.csv",
            "runs/comparison_results.csv"
        ]
        png_files = glob.glob("runs/*.png")

        for file_pattern in files_to_check:
            exists = "✓" if os.path.exists(file_pattern) else "✗"
            print(f"  {exists} {file_pattern}")

        for png_file in png_files:
            print(f"  ✓ {png_file}")

        if not any(os.path.exists(f) for f in files_to_check) and not png_files:
            print("🎯 Nothing to clean")
    else:
        clean_runs()


if __name__ == "__main__":
    main()