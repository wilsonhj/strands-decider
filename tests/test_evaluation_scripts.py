"""The evaluation scripts keep pace with the library they probe.

CI lints `evaluation/` (ruff) but neither type-checks nor runs it, and the scripts need a
GPU or the Hub, so a removed API can survive in them for a release: the slot-head removal
left `frozen_slot_log_probs`, `slot_token_ids` and `CollatorConfig(num_slots=...)` calls in
two of them. These checks run on a CPU with nothing downloaded:

  - every script byte-compiles;
  - every name a script imports from `strands_decider` or `training` still exists there;
  - every keyword a script passes to a config dataclass is still a field of it;
  - no script calls a method or reads a field that the slot-head removal deleted.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import py_compile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = sorted((ROOT / "evaluation").rglob("*.py"))

# Dataclasses whose constructor calls are checked, by the name a script imports them under.
CONFIGS = {
    "CollatorConfig": "training.data.collate",
    "StrandsDeciderConfig": "strands_decider.modeling",
    "TrainConfig": "training.train",
}
# Attributes and names the slot-head removal (13c0514) deleted; none may reappear in a script.
REMOVED = {
    "DEFAULT_NUM_SLOTS",
    "SlotHead",
    "_output_embedding",
    "frozen_slot_log_probs",
    "head_hidden",
    "head_init",
    "init_head_from_lm_head",
    "num_slots",
    "slot_token_ids",
}


def _id(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_the_scripts_are_found():
    assert len(SCRIPTS) > 10, SCRIPTS


@pytest.mark.parametrize("path", SCRIPTS, ids=_id)
def test_script_compiles(path, tmp_path):
    py_compile.compile(str(path), cfile=str(tmp_path / "out.pyc"), doraise=True)


@pytest.mark.parametrize("path", SCRIPTS, ids=_id)
def test_imports_from_the_library_resolve(path):
    for node in ast.walk(_tree(path)):
        if not (isinstance(node, ast.ImportFrom) and node.level == 0 and node.module):
            continue
        if node.module.split(".")[0] not in ("strands_decider", "training"):
            continue
        mod = importlib.import_module(node.module)
        for alias in node.names:
            if alias.name == "*" or hasattr(mod, alias.name):
                continue
            try:  # `from pkg import submodule`
                importlib.import_module(f"{node.module}.{alias.name}")
            except ImportError:
                pytest.fail(f"{_id(path)}:{node.lineno}: {node.module} has no {alias.name!r}")


@pytest.mark.parametrize("path", SCRIPTS, ids=_id)
def test_config_keywords_are_fields(path):
    for node in ast.walk(_tree(path)):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", getattr(node.func, "attr", None))
        if name not in CONFIGS:
            continue
        fields = {
            f.name
            for f in dataclasses.fields(getattr(importlib.import_module(CONFIGS[name]), name))
        }
        for kw in node.keywords:
            if kw.arg is not None and kw.arg not in fields:
                pytest.fail(f"{_id(path)}:{node.lineno}: {name} has no field {kw.arg!r}")


@pytest.mark.parametrize("path", SCRIPTS, ids=_id)
def test_no_slot_head_api_is_used(path):
    for node in ast.walk(_tree(path)):
        used = (
            node.attr
            if isinstance(node, ast.Attribute)
            else node.id
            if isinstance(node, ast.Name)
            else node.arg
            if isinstance(node, ast.keyword)
            else node.name
            if isinstance(node, ast.alias)
            else None
        )
        if used in REMOVED:
            pytest.fail(f"{_id(path)}:{getattr(node, 'lineno', '?')}: {used!r} was removed")
