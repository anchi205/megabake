"""The pinned CUTLASS upstream content next to the GEMM bodies is hash-checked."""

from __future__ import annotations

import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_cutlass_upstream_copy_is_unmodified():
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "vendor_cutlass.py"), "--check"],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout


@pytest.mark.parametrize("modify_header", [False, True])
def test_sm100_upstream_hash_check_rejects_modified_content(monkeypatch, tmp_path, modify_header):
    spec = spec_from_file_location("vendor_cutlass", ROOT / "tools" / "vendor_cutlass.py")
    vendor = module_from_spec(spec)
    spec.loader.exec_module(vendor)
    modified = tmp_path / "fp16_gemm_0.py"
    reference = next(item for item in vendor.REFERENCES if item["header"])
    source = reference["original"].read_bytes()
    modified.write_bytes(b"# changed\n" + source if modify_header else source + b"\n# changed\n")
    monkeypatch.setitem(reference, "original", modified)
    with pytest.raises(SystemExit, match="upstream .* modified"):
        vendor.check_upstream()
