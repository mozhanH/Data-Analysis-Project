"""Local VS Code configuration shared by Phase 1 and Phase 2 notebooks.

Edit config/local_settings.json (NOT the notebooks) to choose data directories.
Any path may be absolute, e.g. D:/Datasets/AmazonReviews or relative to this project.
No downloading is performed by this module or either notebook.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class LocalSettings:
    project_root: Path
    search_roots: tuple[Path, ...]
    review_paths: dict[str, Optional[Path]]
    metadata_paths: dict[str, Optional[Path]]
    output_root: Path
    categories: list[str]
    sample_size_per_category: int
    seed: int


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the project root from VS Code root or notebooks/ working directory."""
    current = Path(start or Path.cwd()).expanduser().resolve()
    for directory in (current, *current.parents):
        if (directory / 'config' / 'local_settings.json').is_file() and (directory / 'src' / '__init__.py').is_file():
            return directory
    raise FileNotFoundError(
        f'Project root not found starting at {current}. Open the extracted '
        'DA_VSCode_Project folder in VS Code and run the notebook from there.'
    )


def _resolve_path(value, root: Path) -> Optional[Path]:
    if value is None or str(value).strip() == '':
        return None
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_local_settings(project_root: Path | None = None) -> LocalSettings:
    root = Path(project_root).resolve() if project_root else find_project_root()
    config_file = root / 'config' / 'local_settings.json'
    settings = json.loads(config_file.read_text(encoding='utf-8'))
    categories = list(settings['categories'])
    roots = tuple(_resolve_path(p, root) for p in settings.get('data_roots', ['data/raw']))
    if not roots:
        raise ValueError('config/local_settings.json: data_roots must not be empty')
    review_paths = {c: _resolve_path(settings.get('review_paths', {}).get(c), root) for c in categories}
    metadata_paths = {c: _resolve_path(settings.get('metadata_paths', {}).get(c), root) for c in categories}
    output_root = _resolve_path(settings.get('output_root', 'outputs'), root)
    if output_root is None:
        raise ValueError('config/local_settings.json: output_root is required')
    if int(settings['sample_size_per_category']) <= 0:
        raise ValueError('sample_size_per_category must be positive')
    return LocalSettings(
        project_root=root,
        search_roots=roots,
        review_paths=review_paths,
        metadata_paths=metadata_paths,
        output_root=output_root,
        categories=categories,
        sample_size_per_category=int(settings['sample_size_per_category']),
        seed=int(settings['seed']),
    )
