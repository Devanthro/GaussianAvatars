#!/usr/bin/env python3
"""Write the avatar.json files that give each avatar its name in the operator app.

``avatar_registry`` reads ``display_name`` from ``<avatar>/avatar.json`` and
falls back to the id otherwise, so without these files the headset shows
legacy ids like ``alona`` for the folder ``lia_mmdm``.  This writes one per
avatar folder, named after the folder (``lia_mmdm`` -> ``lia_mmdm``) or with
the name given with ``--set``::

    python3 write_avatar_json.py --dry-run                 # show what would be written
    python3 write_avatar_json.py                           # all avatars missing one
    python3 write_avatar_json.py --set mubera_caritas="Mubera Ç." --set flo=Florian
    python3 write_avatar_json.py --force                   # rewrite existing ones too

Other keys already in an avatar.json are kept.  The running node picks the
files up within seconds and publishes a new revision.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from avatar_registry import DEFAULT_AVATAR_ROOT, METADATA_FILENAME
from render_thumbnails import avatar_folders


def default_display_name(folder_name: str) -> str:
    """The folder name as it is, until the avatars get proper names."""
    return folder_name


def parse_overrides(pairs: Optional[List[str]]) -> Dict[str, str]:
    overrides = {}
    for pair in pairs or []:
        folder, sep, name = pair.partition("=")
        if not sep or not folder.strip() or not name.strip():
            raise ValueError(f"--set expects FOLDER=NAME, got {pair!r}")
        overrides[folder.strip()] = name.strip()
    return overrides


def read_metadata(path: Path) -> dict:
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    if not isinstance(meta, dict):
        raise ValueError(f"{path} does not hold a JSON object")
    return meta


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Write avatar.json (display_name) for the operator app's avatar picker.")
    parser.add_argument("--root", type=Path, default=Path(DEFAULT_AVATAR_ROOT),
                        help=f"avatar directory (default: {DEFAULT_AVATAR_ROOT})")
    parser.add_argument("--avatar", action="append", metavar="NAME",
                        help="only this avatar folder; repeatable")
    parser.add_argument("--set", action="append", metavar="FOLDER=NAME",
                        help="display name for one folder instead of the derived one; repeatable")
    parser.add_argument("--force", action="store_true",
                        help="overwrite the display_name of avatars that already have one")
    parser.add_argument("--dry-run", action="store_true",
                        help="only report what would be written")
    args = parser.parse_args(argv)

    if not args.root.is_dir():
        print(f"error: no such avatar directory: {args.root}", file=sys.stderr)
        return 2
    try:
        overrides = parse_overrides(args.set)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    folders = avatar_folders(args.root, args.avatar)
    unknown = sorted(set(overrides) - {f.name for f in folders})
    if unknown:
        print(f"error: --set for folders that are not avatars here: {', '.join(unknown)}", file=sys.stderr)
        return 2
    if not folders:
        print(f"no avatars found in {args.root}")
        return 0

    written = skipped = failed = 0
    final_names: Dict[str, List[str]] = {}
    for folder in folders:
        if not folder.is_dir():
            print(f"  {folder.name:<20} missing folder")
            failed += 1
            continue

        path = folder / METADATA_FILENAME
        try:
            meta = read_metadata(path)
        except (OSError, ValueError) as exc:
            print(f"  {folder.name:<20} unreadable {METADATA_FILENAME}: {exc}")
            failed += 1
            continue

        current = meta.get("display_name")
        name = overrides.get(folder.name, default_display_name(folder.name))
        # An explicit --set always wins; a derived name never replaces one someone chose.
        keep = current and folder.name not in overrides and not args.force
        final_names.setdefault(current if keep else name, []).append(folder.name)
        if keep:
            print(f"  {folder.name:<20} has \"{current}\" already (use --force or --set to replace)")
            skipped += 1
            continue
        if current == name:
            print(f"  {folder.name:<20} \"{name}\" unchanged")
            skipped += 1
            continue
        if args.dry_run:
            print(f"  {folder.name:<20} would write \"{name}\"")
            continue

        meta["display_name"] = name
        try:
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            print(f"  {folder.name:<20} failed: {exc}")
            failed += 1
            continue

        written += 1
        print(f"  {folder.name:<20} -> \"{name}\"")

    # The picker shows only the name, so two avatars with the same one look identical.
    for name, names in final_names.items():
        if len(names) > 1:
            print(f"warning: \"{name}\" is the name of {', '.join(names)}; tell them apart with --set")

    print(f"{written} written, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
