"""Check the VS Code local configuration without downloading or parsing full datasets.

Run from the project root: python -m scripts.check_setup
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.project_paths import load_local_settings
from src.data_io import find_jsonl_file


def main():
    cfg = load_local_settings(ROOT)
    print("Project root:", cfg.project_root)
    print("Search folders:", *(str(p) for p in cfg.search_roots), sep="\n  - ")
    print("Expected sample per category:", cfg.sample_size_per_category)
    any_missing = False
    for category in cfg.categories:
        print(f"\n[{category}]")
        for metadata, configured in ((False, cfg.review_paths[category]), (True, cfg.metadata_paths[category])):
            role = "metadata" if metadata else "reviews"
            path = find_jsonl_file(category, cfg.search_roots, metadata=metadata, manual_path=configured)
            if path is None:
                any_missing = True
                print(f"  {role}: NOT FOUND")
            else:
                print(f"  {role}: {path}")
    if any_missing:
        print("\nOne or more required input files were not found. Set data_roots or explicit paths in")
        print("config/local_settings.json. Files can remain compressed as .jsonl.gz.")
        raise SystemExit(2)
    print("\nSetup check PASSED. Both notebooks can access the configured input files.")


if __name__ == '__main__':
    main()
