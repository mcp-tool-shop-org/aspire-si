"""check_apply_item (gates 1.7.5, R&D ruling 2026-10-17): apply items get check_item's content and shape checks, without
its never-trained refusal, plus the apply rules; and neither door lets an apply item into training."""

import copy
import json
from pathlib import Path

import pytest

import aspire_si_gates as G
from aspire_si_gates.gates import DataError

FIX = Path(__file__).parent / "fixtures"


def apply_row(kind: str = "D") -> dict:
    """A T04 apply item that passes: a role-set mock row (fixture, mock text only) turned into an apply item."""
    return json.loads((FIX / f"apply-{kind}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("kind", ["C", "D"])
def test_a_valid_apply_item_passes(kind):
    G.check_apply_item(apply_row(kind))


def _refused(row: dict, match: str) -> None:
    with pytest.raises(DataError, match=match):
        G.check_apply_item(row)


def test_only_an_apply_item_is_taken():
    r = apply_row()
    trained = copy.deepcopy(r)
    trained.pop("split")
    _refused(trained, "takes only apply items")
    no_ap = copy.deepcopy(r)
    no_ap["id"] = no_ap["id"].replace("-AP-", "-")
    _refused(no_ap, "takes only apply items")


def test_the_apply_rules_are_enforced():
    r = apply_row()
    kind_b = dict(copy.deepcopy(r), kind="B")
    _refused(kind_b, "kind C or D")
    two = copy.deepcopy(r)
    two["turns"] = two["turns"] * 2
    _refused(two, "one turn")
    keyed = dict(copy.deepcopy(r), turn_keys={"1": {}})
    _refused(keyed, "one turn")
    paired = dict(copy.deepcopy(r), pair="t04-pair-1")
    _refused(paired, "no contrast pair")


def test_the_silent_twin_flag():
    r = apply_row()
    missing = copy.deepcopy(r)
    missing.pop("silent_without_deciding")
    _refused(missing, "needs silent_without_deciding")
    wrong_type = dict(copy.deepcopy(r), silent_without_deciding="yes")
    _refused(wrong_type, "needs silent_without_deciding")
    silent_ct = dict(copy.deepcopy(r), verdict="cannot_tell", silent_without_deciding=True)
    _refused(silent_ct, "never true on a cannot_tell item")


def test_the_t09_and_t10_twin_fields():
    t09 = dict(copy.deepcopy(apply_row()), lesson="T09")
    t09["id"] = t09["id"].replace("T04-", "T09-")
    t09.pop("silent_without_deciding")
    _refused(t09, "T09 apply item needs order_free")
    t10 = dict(copy.deepcopy(apply_row()), lesson="T10")
    t10["id"] = t10["id"].replace("T04-", "T10-")
    t10.pop("silent_without_deciding")
    _refused(t10, "needs r10_twin")
    same = dict(t10, r10_twin={"statement": "the depot opens every day.", "verdict": t10["verdict"]})
    _refused(same, "must differ")


def test_neither_door_lets_an_apply_item_into_training():
    """Regressions (R&D): check_item and validate_training_set still refuse apply rows."""
    r = apply_row()
    with pytest.raises(DataError, match="never trained"):
        G.check_item(r)
    with pytest.raises(DataError, match="never trained"):
        G.validate([r])
