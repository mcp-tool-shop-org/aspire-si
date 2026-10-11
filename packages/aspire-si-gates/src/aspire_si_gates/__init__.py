"""aspire-si-gates: the Stage 1 data gates, shared by the trainer (aspire-si) and the dataset generator (role-set).

    from aspire_si_gates import validate, check_item, check_against, check_batch, gates_identity
"""

import hashlib
from pathlib import Path

from .context import REASONS, check_against  # noqa: F401
from .scorer_rules import critic_mismatch, deciding_covers, deciding_ok, lines_of, norm  # noqa: F401
from .gates import (DataError, batch_report, check_batch, check_item, closed_form_options,  # noqa: F401
                    freeze_report, leakage, load, parse_choice, role_words_in_prompt, validate,
                    validate_training_set, CLOSED_FORM_DEFAULT, closed_form_settings,
                    key_point_share, closed_form_figures)

GATES_VERSION = "1.4.0"


def gates_identity() -> dict:
    """{"version", "sha256"}: the sha256 over this package's .py files (name and bytes, in sorted order), so a
    generator can record exactly which gates passed an item."""
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob("*.py")):
        h.update(f.name.encode() + b"\0" + f.read_bytes() + b"\0")
    return {"version": GATES_VERSION, "sha256": h.hexdigest()}
