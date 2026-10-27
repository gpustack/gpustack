"""A soft delete has to reach a ``deleted_at IS NULL`` watch stream.

``streaming(fields={"deleted_at": None})`` means "stream the rows that are
live". Callers read that as a query filter, but ``_match_fields`` re-applies it
to every bus event — and a soft delete is published with ``deleted_at`` already
set (``ActiveRecordMixin.delete`` registers the DELETED event, then stamps the
column), so the one event that says "this row is gone" was the one being
dropped. An open page kept rendering the row until a reload.

``GPUInstanceType`` stands in for every model with the shape; the fix is in the
mixin, so ``clusters``, ``cloud_credentials``, ``model_provider`` and
``organizations`` inherit it.
"""

import json
from datetime import datetime

import pytest

from gpustack.schemas.gpu_instance_types import GPUInstanceType, GPUInstanceTypeSpec
from gpustack.server.bus import Event, EventType

RETIRED_AT = datetime(2020, 1, 1)


def _row(*, cluster_id=1, name="a10g", deleted_at=None):
    row = GPUInstanceType(
        cluster_id=cluster_id,
        name=name,
        spec=GPUInstanceTypeSpec.model_validate({"unitResources": {"ram": "1Mi"}}),
    )
    row.snapshot = row.compute_snapshot()
    row.id = 7
    row.deleted_at = deleted_at
    return row


def _event(event_type, row):
    return Event(type=event_type, data=row)


LIVE_ROWS_ONLY = {"deleted_at": None}


def test_a_soft_delete_reaches_a_live_rows_only_stream():
    # The regression: this returned False, so the client was never told.
    event = _event(EventType.DELETED, _row(deleted_at=RETIRED_AT))
    assert GPUInstanceType._match_fields(event, LIVE_ROWS_ONLY) is True


def test_a_live_row_still_reaches_it():
    # Positive control — the exemption must not be doing all the work.
    event = _event(EventType.CREATED, _row())
    assert GPUInstanceType._match_fields(event, LIVE_ROWS_ONLY) is True


def test_an_update_that_sets_deleted_at_is_still_filtered():
    # Only DELETED is exempt. An UPDATED carrying a set deleted_at is not the
    # "row is gone" notification and stays filtered, as before.
    event = _event(EventType.UPDATED, _row(deleted_at=RETIRED_AT))
    assert GPUInstanceType._match_fields(event, LIVE_ROWS_ONLY) is False


def test_a_soft_delete_outside_the_streams_cluster_is_still_rejected():
    # The exemption is per-key, not per-event: a stream scoped to one cluster
    # must not learn about another cluster's deletions. Exempting the whole
    # fields match on DELETED would leak them.
    event = _event(EventType.DELETED, _row(cluster_id=5, deleted_at=RETIRED_AT))
    assert (
        GPUInstanceType._match_fields(event, {"deleted_at": None, "cluster_id": 3})
        is False
    )


def test_a_soft_delete_under_a_name_filter_still_matches_only_that_name():
    event = _event(EventType.DELETED, _row(name="a10g", deleted_at=RETIRED_AT))
    assert (
        GPUInstanceType._match_fields(event, {"deleted_at": None, "name": "l40s"})
        is False
    )


def test_the_exemption_only_applies_when_the_filter_asks_for_live_rows():
    # ``deleted_at: <timestamp>`` is nobody's filter today, but it would mean
    # "this exact retirement", not "live rows" — so it keeps matching literally
    # rather than waving every deletion through.
    event = _event(EventType.DELETED, _row(deleted_at=RETIRED_AT))
    assert (
        GPUInstanceType._match_fields(event, {"deleted_at": datetime(1999, 1, 1)})
        is False
    )


@pytest.mark.asyncio
async def test_a_definition_change_streams_the_retirement_then_the_replacement(
    monkeypatch,
):
    """End to end over the real ``streaming()``, at the level a client sees.

    A definition change makes the controller retire the active row and insert a
    new one under the same name. Before the fix only the CREATED frame arrived,
    so an open page rendered two rows under one name until reload.
    """
    retired = _row(deleted_at=RETIRED_AT)
    replacement = _row()
    replacement.id = 8

    async def fake_subscribe(*args, **kwargs):
        yield _event(EventType.DELETED, retired)
        yield _event(EventType.CREATED, replacement)

    monkeypatch.setattr(GPUInstanceType, "subscribe", fake_subscribe)

    frames = [
        json.loads(frame)
        async for frame in GPUInstanceType.streaming(fields=LIVE_ROWS_ONLY)
    ]

    # The wire carries EventType's value, not its name: CREATED=1, DELETED=3.
    assert [f["type"] for f in frames] == [
        EventType.DELETED.value,
        EventType.CREATED.value,
    ]
    assert [f["data"]["id"] for f in frames] == [7, 8]
