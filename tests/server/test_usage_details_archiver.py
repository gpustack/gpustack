"""Pure unit tests for ``UsageDetailsArchiver``.

DB-level (sqlite/postgres) integration is intentionally out of scope here.
This file covers everything that doesn't touch the database:
  * calendar arithmetic for the retention cutoff
  * hot ↔ archive table shape alignment (required by the bulk SQL path)
  * cron expression validation + next-fire computation
"""

from datetime import datetime
import pytest

from gpustack.schemas.model_usage_details import (
    ModelUsageDetails,
    ModelUsageDetailsArchive,
)
from gpustack.server.usage_details_archiver import (
    UsageDetailsArchiver,
    _assert_archive_shape_aligned,
    _months_ago,
)


# ---------------------------------------------------------------------------
# _months_ago — calendar arithmetic, not timedelta(days=30*N)
# ---------------------------------------------------------------------------


def test_months_ago_simple_subtraction():
    assert _months_ago(datetime(2026, 5, 7, 12, 0), 13) == datetime(2025, 4, 7, 12, 0)


def test_months_ago_clamps_short_target_month():
    # 3/31 - 1 month must clamp to 2/28 (or 2/29 in leap years), not overflow to 3/3.
    assert _months_ago(datetime(2026, 3, 31, 12, 0), 1) == datetime(2026, 2, 28, 12, 0)


def test_months_ago_handles_leap_year():
    assert _months_ago(datetime(2024, 3, 31), 1) == datetime(2024, 2, 29)


def test_months_ago_wraps_year_boundary():
    assert _months_ago(datetime(2026, 1, 15, 12, 0), 2) == datetime(2025, 11, 15, 12, 0)


def test_months_ago_wraps_year_boundary_multi():
    # 25 months back crosses two year boundaries.
    assert _months_ago(datetime(2026, 5, 7), 25) == datetime(2024, 4, 7)


def test_months_ago_preserves_time_of_day():
    src = datetime(2026, 5, 7, 13, 45, 30, 123456)
    out = _months_ago(src, 13)
    assert (out.hour, out.minute, out.second, out.microsecond) == (13, 45, 30, 123456)


# ---------------------------------------------------------------------------
# Hot ↔ archive table shape alignment
# (bulk INSERT ... SELECT positionally requires identical column lists)
# ---------------------------------------------------------------------------


def test_archive_shape_alignment_passes_for_current_schemas():
    # Should not raise — this is the runtime contract the archiver depends on.
    _assert_archive_shape_aligned()


def test_archive_shape_alignment_includes_all_business_columns():
    """Belt-and-suspenders against silent column loss — call out the columns
    we expect on both sides explicitly so a future schema change can't strip
    them and still pass ``_assert_archive_shape_aligned``."""
    expected = {
        "id",
        "user_id",
        "user_name",
        "model_id",
        "model_name",
        "model_route_id",
        "model_route_name",
        "provider_id",
        "provider_name",
        "provider_type",
        "cluster_id",
        "cluster_name",
        "api_key_id",
        "api_key_name",
        "access_key",
        "api_key_is_custom",
        "date",
        "prompt_token_count",
        "completion_token_count",
        "prompt_cached_token_count",
        "operation",
        "started_at",
        "completed_at",
        "created_at",
        "updated_at",
        "deleted_at",
    }
    hot = {c.name for c in ModelUsageDetails.__table__.columns}
    archive = {c.name for c in ModelUsageDetailsArchive.__table__.columns}
    assert expected <= hot, f"hot table missing: {expected - hot}"
    assert expected <= archive, f"archive table missing: {expected - archive}"


def test_archive_shape_alignment_raises_on_drift(monkeypatch):
    """Synthetic drift via monkeypatched __table__ proves the assertion
    actually trips — without this we can't trust the no-raise case above."""

    class _FakeColumn:
        def __init__(self, name):
            self.name = name

    class _FakeTable:
        def __init__(self, columns):
            self.columns = columns

    drifted_archive = _FakeTable(
        [_FakeColumn(c.name) for c in ModelUsageDetailsArchive.__table__.columns]
    )
    drifted_archive.columns = drifted_archive.columns[:-1]  # drop one column
    monkeypatch.setattr(ModelUsageDetailsArchive, "__table__", drifted_archive)
    with pytest.raises(RuntimeError, match="column mismatch"):
        _assert_archive_shape_aligned()


# ---------------------------------------------------------------------------
# Construction-time validation of the cron expression
# ---------------------------------------------------------------------------


def test_archiver_rejects_invalid_cron(monkeypatch):
    monkeypatch.setattr("gpustack.envs.USAGE_DETAILS_ARCHIVE_CRON", "garbage")
    with pytest.raises(ValueError, match="USAGE_DETAILS_ARCHIVE_CRON"):
        UsageDetailsArchiver()


def test_archiver_rejects_empty_cron(monkeypatch):
    monkeypatch.setattr("gpustack.envs.USAGE_DETAILS_ARCHIVE_CRON", "")
    with pytest.raises(ValueError, match="USAGE_DETAILS_ARCHIVE_CRON"):
        UsageDetailsArchiver()


def test_archiver_reads_retention_and_batch_size(monkeypatch):
    monkeypatch.setattr("gpustack.envs.USAGE_DETAILS_RETENTION_MONTHS", 7)
    monkeypatch.setattr("gpustack.envs.USAGE_DETAILS_ARCHIVE_BATCH_SIZE", 250)
    arc = UsageDetailsArchiver()
    assert arc._retention_months == 7
    assert arc._batch_size == 250


# ---------------------------------------------------------------------------
# Next-fire computation across common cron expressions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr,upper_bound_seconds",
    [
        ("0 3 * * *", 24 * 3600),  # daily at 03:00 — within a day
        ("*/15 * * * *", 15 * 60),  # every 15 minutes
        ("0 */6 * * *", 6 * 3600),  # every 6 hours
        ("30 2 * * 0", 7 * 24 * 3600),  # weekly Sunday — within a week
        ("0 0 1 * *", 32 * 24 * 3600),  # 1st of month — within ~1 month
    ],
)
def test_seconds_until_next_fire_within_bound(monkeypatch, expr, upper_bound_seconds):
    monkeypatch.setattr("gpustack.envs.USAGE_DETAILS_ARCHIVE_CRON", expr)
    secs = UsageDetailsArchiver()._seconds_until_next_fire()
    assert secs is not None
    assert 0 < secs <= upper_bound_seconds


def test_seconds_until_next_fire_returns_float(monkeypatch):
    monkeypatch.setattr("gpustack.envs.USAGE_DETAILS_ARCHIVE_CRON", "*/5 * * * *")
    assert isinstance(UsageDetailsArchiver()._seconds_until_next_fire(), float)
