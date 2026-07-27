import os
import random
import tarfile
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:  # pragma: no cover
    hf_hub_download = None

"""Simple teams loader for Gen3 OU packs.

The loader can read local team packs from data/teams/<formatid>.txt, or
fetch a HuggingFace-hosted Metamon team set if no local file exists.

Functions:
- load_team_packs(formatid)
- get_random_team(formatid)
- get_team_by_index(formatid, index)
- download_metamon_teams(formatid, set_name='competitive')
"""

METAMON_TEAMS_REPO = "jakegrigsby/metamon-teams"
METAMON_TEAMS_VERSION = "v5"
METAMON_TEAM_SET = "competitive"


def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))


def _teams_dir():
    return os.path.join(_repo_root(), 'data', 'teams')


def _team_cache_dir():
    return os.path.join(_repo_root(), 'data', 'teams_cache')


def _convert_to_multiline(pack):
    """Convert a Showdown team pack from single-line compressed format
    (fields separated by '   / ') to multi-line format that Teams.import
    accepts (newlines between fields, blank line between Pokémon).

    The single-line format was produced by download_metamon_teams.py
    replacing every '\\n' with ' / ', which makes Teams.unpack fail and
    Teams.import only see the first Pokémon.
    """
    # Separator between Pokémon: "   /  " (3 spaces + / + 2 spaces)
    # Separator between fields: "   / " (3 spaces + / + 1 space)
    # Order matters: replace Pokémon separator first (it's more specific).
    pack = pack.replace("   /  ", "\n\n")
    pack = pack.replace("   / ", "\n")
    # After conversion, the 2nd+ Pokémon start with "/ " left over from the
    # original " /  / " separator — strip that leading "/ " from each line.
    lines = pack.split("\n")
    cleaned = [line[2:] if line.startswith("/ ") else line for line in lines]
    return "\n".join(cleaned).strip()


def _load_packs_from_file(path):
    packs = []
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            s = line.strip()
            if s:
                packs.append(_convert_to_multiline(s))
    return packs


def _load_packs_from_dir(path):
    packs = []
    for root, _, files in os.walk(path):
        for fname in files:
            if fname.startswith('.'):
                continue
            if fname.lower().endswith(('.txt', '.pack', '.team')):
                packs.extend(_load_packs_from_file(os.path.join(root, fname)))
            else:
                candidate_path = os.path.join(root, fname)
                if os.path.getsize(candidate_path) > 100_000:
                    continue
                try:
                    for line in _load_packs_from_file(candidate_path):
                        if '|' in line:
                            packs.append(line)
                except UnicodeDecodeError:
                    continue
    return packs


def download_metamon_teams(formatid='gen3ou', set_name=METAMON_TEAM_SET, force_download=False):
    """Download a Metamon team set from HuggingFace and return the extracted directory."""
    if hf_hub_download is None:
        raise RuntimeError(
            "huggingface_hub is required to download Metamon teams. "
            "Install it with `pip install huggingface-hub`."
        )

    cache_dir = os.path.abspath(os.path.join(_team_cache_dir(), set_name))
    os.makedirs(cache_dir, exist_ok=True)
    tar_name = f"{formatid}.tar.gz"
    tar_path = os.path.join(cache_dir, tar_name)
    extract_dir = os.path.join(cache_dir, formatid)

    if os.path.isdir(extract_dir) and not force_download:
        return extract_dir

    if os.path.exists(extract_dir) and force_download:
        for root, dirs, files in os.walk(extract_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(extract_dir)

    hf_hub_download(
        cache_dir=cache_dir,
        repo_id=METAMON_TEAMS_REPO,
        filename=f"{set_name}/{tar_name}",
        local_dir=cache_dir,
        revision=METAMON_TEAMS_VERSION,
        repo_type="dataset",
    )

    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=cache_dir)
    os.remove(tar_path)
    return extract_dir


def load_team_packs(formatid='gen3ou'):
    """Load packed teams for a format. Returns list of pack strings."""
    teams_path = _teams_dir()
    os.makedirs(teams_path, exist_ok=True)
    # Prefer a local format-specific pack file before scanning the teams directory
    candidates = [
        os.path.join(teams_path, f"{formatid}.txt"),
        os.path.join(teams_path, f"{formatid}.pack"),
    ]
    packs = []
    for path in candidates:
        if os.path.isfile(path):
            packs.extend(_load_packs_from_file(path))
            if packs:
                return packs

    packs = _load_packs_from_dir(teams_path)
    if packs:
        return packs

    try:
        download_dir = download_metamon_teams(formatid=formatid)
        packs = _load_packs_from_dir(download_dir)
    except Exception:
        packs = []
    return packs


def get_random_team(formatid='gen3ou'):
    packs = load_team_packs(formatid)
    if not packs:
        return None
    return random.choice(packs)


def get_team_by_index(formatid='gen3ou', index=0):
    packs = load_team_packs(formatid)
    if not packs:
        return None
    if index < 0 or index >= len(packs):
        return None
    return packs[index]
