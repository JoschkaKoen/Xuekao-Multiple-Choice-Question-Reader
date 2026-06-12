"""Load the student roster from name_list.xlsx (Sheet1, column 姓名)."""

from __future__ import annotations

import logging
from pathlib import Path

import openpyxl

log = logging.getLogger("xuekao")


def load_roster(path: Path) -> list[str]:
    """Return the list of student names (col B / 姓名), de-duplicated, in order."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    names: list[str] = []
    seen: set[str] = set()
    for row in ws.iter_rows(values_only=True):
        if len(row) < 2 or row[1] is None:
            continue
        s = str(row[1]).strip()
        if not s or s == "姓名":  # skip blanks + the header cell
            continue
        if s not in seen:
            seen.add(s)
            names.append(s)
    wb.close()
    log.info("roster: %d names from %s", len(names), path.name)
    return names
