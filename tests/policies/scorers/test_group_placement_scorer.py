"""Ranking one whole-group placement against another.

The property these pin down is that a group is scored as an *assignment*, not
as a bag of members. Pairing is the clearest case: the objective is
``sum_j p_j*d_j / (x*y)``, and adding up what each member would score on its
own counts every pair twice -- once from the prefill's end and once from the
decode's -- which ranks a different question.
"""

import pytest

from gpustack.policies.scorers.group_placement_scorer import (
    file_locality,
    group_scorer,
    pair_locality,
)
from gpustack.scheduler.group_solver import GroupPlacement


def _placement(**assignments) -> GroupPlacement:
    return GroupPlacement(layer="Rack", path=["rack-a"], assignments=assignments)


@pytest.mark.parametrize(
    "prefill, decode, expected, why",
    [
        ([1], [1], 1.0, "the only pair is on one host"),
        ([1], [2], 0.0, "the only pair is split"),
        ([1, 2], [1, 2], 0.5, "two of four possible pairings are local"),
        ([1, 1], [1, 1], 1.0, "everything on one host, every pairing local"),
        ([1, 2], [3, 4], 0.0, "no host holds both roles"),
        ([1, 1, 2], [1], 2 / 3, "two of three prefills share the decode's host"),
    ],
)
def test_pair_locality_is_the_share_of_local_pairings(prefill, decode, expected, why):
    got = pair_locality(_placement(prefill=prefill, decode=decode))
    assert got == pytest.approx(expected), why


def test_a_role_with_no_opposite_scores_zero_rather_than_dividing_by_it():
    """A group with only one paired role has no pair to keep local. Returning
    1.0 would make it outrank every real group; dividing would raise."""
    assert pair_locality(_placement(prefill=[1, 2])) == 0.0
    assert pair_locality(_placement(decode=[1, 2])) == 0.0
    assert pair_locality(_placement()) == 0.0


def test_pairing_is_a_ratio_and_not_a_count_of_siblings():
    """Why the objective and not the per-member sum. Summing what each member
    would score on its own comes to exactly twice this -- so it ranks the
    same, and is still the wrong quantity to hand back: it grows with the
    group's size and with the per-candidate scorer's own scale, so adding it
    to a ratio would turn the weight beside it into an exchange rate."""
    small = _placement(prefill=[1], decode=[1])
    large = _placement(prefill=[1, 2, 3, 4], decode=[1, 2, 3, 4])

    # Both pair as well as their size allows, and both say so with the same
    # number. A sibling count would rank the larger group four times better.
    assert pair_locality(small) == 0.25 * 4
    assert pair_locality(large) == 0.25
    assert pair_locality(_placement(prefill=[1, 1], decode=[1, 1])) == 1.0


@pytest.mark.parametrize(
    "workers, ready, expected",
    [
        ([1, 1, 2], {1, 2}, 1.0),
        ([1, 1, 2], {1}, 2 / 3),
        ([1, 1, 2], {3}, 0.0),
        ([1, 1, 2], set(), 0.0),
    ],
)
def test_file_locality_is_the_share_of_members_landing_warm(workers, ready, expected):
    got = file_locality(_placement(prefill=workers), ready)
    assert got == pytest.approx(expected)


def test_the_score_carries_the_terms_that_produced_it():
    """A total alone cannot say why one domain beat another -- "paired badly
    but warm" and "paired well and cold" can reach the same number. The log
    line and anything downstream read these rather than evaluating the terms a
    second time against the same placement."""
    score = group_scorer(ready_worker_ids={1})
    got = score(_placement(prefill=[1], decode=[1]))

    assert got.terms == {"pair": 1.0, "file": 1.0}
    assert got.total == pytest.approx(1.0 + 0.3)
    assert got.describe() == "pair 1.00, file 1.00"


def test_a_scorer_with_nothing_to_say_is_none_rather_than_a_constant():
    """The solver reads `score is None` as "take the tightest fitting domain"
    and stops enumerating a layer after the first fit. A scorer that ranks
    everything equally would buy a full selector sweep per extra domain to
    reach the same answer."""
    assert group_scorer(ready_worker_ids=(), pair_weight=0, file_weight=0.3) is None
    assert group_scorer(ready_worker_ids={1}, pair_weight=0, file_weight=0) is None
    assert group_scorer(ready_worker_ids={1}, pair_weight=0, file_weight=0.3)


def test_file_locality_can_only_settle_a_pairing_tie():
    """Co-locating a request's two ends is what disaggregation is for; landing
    warm saves a download once. The default weights have to say so."""
    score = group_scorer(ready_worker_ids={2})
    paired_cold = _placement(prefill=[1], decode=[1])
    unpaired_warm = _placement(prefill=[2], decode=[3])

    assert score(paired_cold).total > score(unpaired_warm).total


def test_file_locality_decides_when_pairing_is_equal():
    score = group_scorer(ready_worker_ids={2})
    warm = _placement(prefill=[2], decode=[2])
    cold = _placement(prefill=[1], decode=[1])

    assert score(warm).total > score(cold).total
    # And the pairing term is identical, so the file term is the whole margin.
    assert pair_locality(warm) == pair_locality(cold)


def _spanned(assignments, spans):
    return GroupPlacement(
        layer="Rack", path=["rack-a"], assignments=assignments, spans=spans
    )


@pytest.mark.parametrize(
    "spans, expected, why",
    [
        ({}, 0.0, "single-machine members on different hosts share nothing"),
        (
            {("prefill", 1): [1, 2]},
            0.5,
            "the decode on host 2 reads half of a prefill spread over 1 and 2",
        ),
        (
            {("prefill", 1): [1, 2, 3, 4]},
            0.25,
            "one of the prefill's four machines carries the decode",
        ),
        (
            {("prefill", 1): [1, 2], ("decode", 2): [2, 5]},
            0.5,
            "the decode's own width does not change what it reads locally",
        ),
    ],
)
def test_a_spanning_prefill_is_local_in_proportion(spans, expected, why):
    """The KV lives across the prefill's machines and the decode reads what
    shares one with it. Measured against the prefill's width, so a decode
    sitting on one of its two hosts reads half -- not none, which is what a
    placement that could only see primaries reported."""
    got = pair_locality(_spanned({"prefill": [1], "decode": [2]}, spans))
    assert got == pytest.approx(expected), why


def test_pairing_reduces_to_the_primary_only_formula_without_spans():
    """The generalisation has to leave every group placed today untouched."""
    plain = _placement(prefill=[1, 2], decode=[1, 2])
    same_but_declared = _spanned(
        {"prefill": [1, 2], "decode": [1, 2]},
        {("prefill", 1): [1], ("decode", 2): [2]},
    )

    assert pair_locality(plain) == pair_locality(same_but_declared) == 0.5


def test_a_spanning_member_is_warm_in_proportion_to_its_machines():
    """It needs the weights on every machine it runs on, so half its hosts
    being warm is half a download saved."""
    half = _spanned({"prefill": [1]}, {("prefill", 1): [1, 2]})
    whole = _spanned({"prefill": [1]}, {("prefill", 1): [1, 2]})

    assert file_locality(half, {1}) == 0.5
    assert file_locality(whole, {1, 2}) == 1.0
    assert file_locality(half, {3}) == 0.0
