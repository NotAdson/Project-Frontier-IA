#!/usr/bin/env python3
"""
Import teams from the Metamon repository (https://github.com/UT-Austin-RPL/metamon)

Usage:
    python scripts/import_metamon_teams.py --repo /path/to/metamon --out data/teams/gen3ou.txt
    python scripts/import_metamon_teams.py --repo https://github.com/UT-Austin-RPL/metamon --out data/teams/gen3ou.txt

The script will try to find packed team strings (Showdown pack format) or JSON
files containing lists of teams and extract/convert them into packed strings.

This is intentionally permissive: Metamon may store teams in different formats,
so the script tries several heuristics. Review the output before using in
experiments.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional


def clone_or_update(repo: str, dest: str) -> str:
    """Clone a git repo URL into dest (temporary dir) or use local path.
    Returns the path to the repo.
    """
    if os.path.isdir(repo):
        return os.path.abspath(repo)
    # treat as URL: clone into dest
    dest_path = os.path.abspath(dest)
    if os.path.exists(dest_path):
        # try to pull
        try:
            subprocess.run(["git", "-C", dest_path, "pull"], check=True)
            return dest_path
        except Exception:
            pass
    # clone
    try:
        print(f"Cloning {repo} into {dest_path}...")
        subprocess.run(["git", "clone", "--depth", "1", repo, dest_path], check=True)
        return dest_path
    except Exception as e:
        raise RuntimeError(f"Failed to clone repo {repo}: {e}")


def find_candidate_files(repo_path: str) -> List[str]:
    candidates = []
    for root, dirs, files in os.walk(repo_path):
        # skip node_modules or .git
        if "node_modules" in root or ".git" in root:
            continue
        for fname in files:
            if fname.lower().endswith(('.txt', '.pack', '.json')) or 'team' in fname.lower() or 'usage' in fname.lower():
                candidates.append(os.path.join(root, fname))
    return candidates


def extract_packs_from_text(path: str) -> List[str]:
    packs = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            # heuristic: a packed team often contains '|' separators
            if '|' in s:
                packs.append(s)
    return packs


def extract_packs_from_json(path: str) -> List[str]:
    packs = []
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
    except Exception:
        return packs

    # If it's a list of strings, assume each string is a pack
    if isinstance(data, list) and data and all(isinstance(x, str) for x in data):
        for s in data:
            if '|' in s:
                packs.append(s.strip())
        return packs

    # If it's a list of objects, try to convert each to a simple pack
    if isinstance(data, list):
        for obj in data:
            if isinstance(obj, dict):
                pack = convert_team_obj_to_pack(obj)
                if pack:
                    packs.append(pack)
    # If it's a dict with keys that may contain teams
    if isinstance(data, dict):
        # common: {"teams": [...]} or {"sets": [...]}
        for key in ('teams', 'sets', 'data'):
            if key in data and isinstance(data[key], list):
                for obj in data[key]:
                    if isinstance(obj, str) and '|' in obj:
                        packs.append(obj.strip())
                    elif isinstance(obj, dict):
                        pack = convert_team_obj_to_pack(obj)
                        if pack:
                            packs.append(pack)
    return packs


def convert_team_obj_to_pack(obj: dict) -> Optional[str]:
    """Try to convert a team object to a packed string. This is heuristic and
    handles common simple structures like [{'species': 'Pikachu', 'item': 'Light Ball', 'moves': [...]}, ...]
    Produces a single-line packed team like: "Pikachu||Light Ball||Ability||move1,move2,move3" per Pokémon separated by '\n' if needed.
    """
    try:
        # If the object itself represents a single Pokemon (has 'species'), not a team
        if 'species' in obj and ('moves' in obj or 'move' in obj):
            # represent single Pokémon as pack fragment
            species = obj.get('species') or obj.get('name')
            nickname = obj.get('nickname', '') or ''
            item = obj.get('item', '') or ''
            ability = obj.get('ability', '') or ''
            moves = obj.get('moves') or obj.get('move') or []
            if isinstance(moves, list):
                moves_s = ','.join(moves)
            else:
                moves_s = str(moves)
            # join as species|nickname|item|ability|moves
            return f"{species}|{nickname}|{item}|{ability}|{moves_s}"

        # If object is a team list: list of pokemon dicts under keys
        # e.g., {'pokemon': [ {...}, {...} ]}
        for key in ('pokemon', 'team', 'set'):
            if key in obj and isinstance(obj[key], list):
                parts = []
                for p in obj[key]:
                    if isinstance(p, dict):
                        species = p.get('species') or p.get('name') or ''
                        nickname = p.get('nickname', '') or ''
                        item = p.get('item', '') or ''
                        ability = p.get('ability', '') or ''
                        moves = p.get('moves') or p.get('move') or []
                        if isinstance(moves, list):
                            moves_s = ','.join(moves)
                        else:
                            moves_s = str(moves)
                        parts.append(f"{species}|{nickname}|{item}|{ability}|{moves_s}")
                if parts:
                    # join pokemon with '\n'
                    return '\n'.join(parts)
    except Exception:
        return None
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True, help='Local path or git URL to metamon repo')
    parser.add_argument('--format', default='gen3ou', help='Format id to extract teams for (default: gen3ou)')
    parser.add_argument('--out', default='data/teams/gen3ou.txt', help='Output file path')
    parser.add_argument('--temp', default=None, help='Temp directory for cloning')
    parser.add_argument('--force', action='store_true', help='Overwrite existing output')
    args = parser.parse_args()

    out_path = os.path.abspath(args.out)
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    # Determine repo path
    repo_arg = args.repo
    tmpdir = args.temp or tempfile.mkdtemp(prefix='metamon_')
    try:
        repo_path = clone_or_update(repo_arg, tmpdir)
    except Exception as e:
        print(f"Failed to obtain repo: {e}")
        sys.exit(1)

    candidates = find_candidate_files(repo_path)
    print(f"Found {len(candidates)} candidate files to scan for team packs...")

    packs = []
    for f in candidates:
        f_lower = f.lower()
        try:
            if f_lower.endswith(('.txt', '.pack')):
                packs.extend(extract_packs_from_text(f))
            elif f_lower.endswith('.json'):
                packs.extend(extract_packs_from_json(f))
            else:
                # Parse as text anyway
                packs.extend(extract_packs_from_text(f))
        except Exception as e:
            print(f"Warning: failed to parse {f}: {e}")

    # Deduplicate and filter
    unique_packs = []
    seen = set()
    for p in packs:
        if not p:
            continue
        norm = p.strip()
        if norm in seen:
            continue
        seen.add(norm)
        unique_packs.append(norm)

    print(f"Extracted {len(unique_packs)} unique candidate packs.")

    if not unique_packs:
        print("No packs found. You may need to run metamon tools to export teams before using this script.")
        sys.exit(1)

    if os.path.exists(out_path) and not args.force:
        print(f"Output file {out_path} already exists. Use --force to overwrite or choose a different --out.")
        sys.exit(1)

    with open(out_path, 'w', encoding='utf-8') as fh:
        for p in unique_packs:
            fh.write(p.replace('\n', ' / ') + '\n')

    print(f"Wrote {len(unique_packs)} packs to {out_path}")


if __name__ == '__main__':
    main()
