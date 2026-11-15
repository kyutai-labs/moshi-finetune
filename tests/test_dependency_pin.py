"""The pip URL must have the exact source identity selected by the lockfile."""

import re
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import finetune

try:
    import tomllib
except ModuleNotFoundError:
    # pytest requires tomli on Python < 3.11, including the supported Python 3.10.
    import tomli as tomllib

ROOT = Path(finetune.__file__).parents[1]


def test_pip_and_uv_select_the_same_explicit_moshi_revision():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    requirement = next(
        value
        for value in project["project"]["dependencies"]
        if value.startswith("moshi @ ")
    )
    url = requirement.split(" @ ", 1)[1]
    pip = urlsplit(url.removeprefix("git+"))
    assert "@" in pip.path, "pip must not follow mutable Git HEAD"
    repo_path, revision = pip.path.rsplit("@", 1)
    assert re.fullmatch(r"[0-9a-f]{40}", revision)
    locked_moshi = next(p for p in lock["package"] if p["name"] == "moshi")
    source = urlsplit(locked_moshi["source"]["git"])
    assert revision == source.fragment
    assert (pip.scheme, pip.netloc, repo_path) == (
        source.scheme,
        source.netloc,
        source.path,
    )
    assert parse_qs(source.query)["rev"] == [revision]
    assert parse_qs(source.query)["subdirectory"] == [pip.fragment.split("=", 1)[1]]
    root = next(p for p in lock["package"] if p["name"] == "finetune")
    metadata = next(
        p for p in root["metadata"]["requires-dist"] if p["name"] == "moshi"
    )
    assert parse_qs(urlsplit(metadata["git"]).query)["rev"] == [revision]
