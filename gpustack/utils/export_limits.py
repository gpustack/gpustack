"""Row-count limits, remedies and file naming for usage exports.

Shared by the token export routes and the resource (GPU / storage) ones, so
the numbers a user is shown before the click are the numbers the download
produces after it: one function decides how many files a split makes, one
decides what to offer when a result is too large, one names the parts.

Nothing here touches the database or the request session — it is arithmetic
over row counts plus the naming conventions the files follow, which is why it
lives outside the routes rather than being imported from one into the other.
"""

from math import ceil, floor
from typing import Any, Dict, List, Optional, Sequence

from gpustack import envs
from gpustack.api.exceptions import InvalidException
from gpustack.schemas.usage import (
    USAGE_EXPORT_FORMAT_CSV,
    USAGE_EXPORT_FORMAT_XLSX,
    USAGE_SCOPE_ALL,
    UsageExportColumn,
    UsageExportEstimateResponse,
    UsageExportSheetEstimate,
)


def attachment_headers(filename: str) -> Dict[str, str]:
    """Response headers that make a browser save the body under ``filename``."""
    return {"Content-Disposition": f'attachment; filename="{filename}"'}


def requested_platform_wide(request) -> bool:
    """Whether the caller ASKED for the platform-wide view.

    ``scope`` defaults to ``all`` so that a manager who omits it gets the
    org-wide page, which means a default ``all`` says nothing about intent: a
    regular user's client sends the same payload. Only an explicitly set
    ``scope`` is a request for cross-user data, and only that may be answered
    with a 403 rather than the usual downgrade to ``self``.
    """
    return "scope" in request.model_fields_set and request.scope == USAGE_SCOPE_ALL


def effective_export_format(requested: str, totals) -> str:
    """The format the response will actually use.

    xlsx is the default because it is what these exports have always
    produced. A worksheet holds at most ``XLSX_MAX_ROWS_PER_SHEET`` rows,
    though, so beyond that the choice is between refusing the export and
    handing back the same data in a format that can hold it. Falling back to
    CSV loses nothing — the rows are identical and the extension announces
    the change — whereas refusing leaves the user with no way to get the data
    at all. The estimate endpoint reports this ahead of the click.
    """
    if requested == USAGE_EXPORT_FORMAT_XLSX and any(
        total >= envs.XLSX_MAX_ROWS_PER_SHEET for total in totals
    ):
        return USAGE_EXPORT_FORMAT_CSV
    return requested


def shorten_range_days(request, total: int, limit: int) -> int:
    """How many days would fit under ``limit`` at the current row density."""
    days = (request.end_date - request.start_date).days + 1
    if total <= 0:
        return days
    return max(1, floor(days * limit / total))


def export_split_plan(totals: Dict[str, int], limit: int) -> Dict[str, int]:
    """How many files each sheet's rows split into: ``ceil(total / limit)``.

    Parts are slices of the ROW STREAM, so this is exact rather than an
    estimate — the count returned here is the count of members the archive
    ends up with. Sheets under the limit still contribute their one file, and
    every sheet contributes, so the archive's member count is this map's sum.

    Shared by the estimate and both split responses so the number offered
    before the click is the number produced after it.
    """
    if limit <= 0:
        return {key: 1 for key in totals}
    return {key: max(1, ceil(total / limit)) for key, total in totals.items()}


def export_suggestions(
    request, total: int, limit: int, split_parts: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Concrete moves that would bring an over-large export under ``limit``.

    Returned by BOTH the estimate endpoint and the over-limit error, from this
    one function: the advice a user sees before clicking and the advice they
    see after a rejection must not contradict each other. Everything here is
    derived from values already in hand, so it costs nothing.

    ``split_parts`` is the file count splitting would actually produce; pass
    ``None`` when the split path would refuse the request (over
    ``USAGE_EXPORT_MAX_SPLIT_MEMBERS``) and the remedy is left out entirely.
    Offering a way out that the next request rejects is worse than offering
    only the one that works.
    """
    # Both remedies are lossless: they change the RANGE or the packaging, never
    # the rows. A coarser granularity would collapse 30 daily rows into one,
    # and sitting next to two lossless options it would be picked without the
    # user noticing the data changed — so it is not offered.
    suggestions: List[Dict[str, Any]] = [
        {
            "action": "shorten_range",
            "max_days": shorten_range_days(request, total, limit),
        }
    ]
    # Offered for every grouping — parts are row slices, so none is excluded —
    # but only when the split path would accept it.
    if split_parts:
        suggestions.append({"action": "split_export", "parts": split_parts})
    return suggestions


def build_export_estimate(
    request, estimates: List[UsageExportSheetEstimate], total: int
) -> UsageExportEstimateResponse:
    """Assemble the estimate response, verdicts included.

    ``over`` is the largest single sheet, because that is the number the
    export routes compare against: sheets are separate queries and separate
    worksheets, so five small tables are not one big one. The FILE count is
    the opposite — every sheet contributes its parts and the cap applies to
    their sum — so the split remedy is withheld when that sum is one the
    export would reject.
    """
    over = max(
        (estimate.total for estimate in estimates if estimate.available), default=0
    )
    exceeds_hard = over > envs.USAGE_EXPORT_MAX_ROWS
    members = sum(
        export_split_plan(
            {
                estimate.key: estimate.total
                for estimate in estimates
                if estimate.available
            },
            envs.USAGE_EXPORT_MAX_ROWS,
        ).values()
    )
    split_parts = (
        members
        if exceeds_hard and members <= envs.USAGE_EXPORT_MAX_SPLIT_MEMBERS
        else None
    )
    return UsageExportEstimateResponse(
        sheets=estimates,
        total=total,
        soft_limit=envs.USAGE_EXPORT_SOFT_ROWS,
        hard_limit=envs.USAGE_EXPORT_MAX_ROWS,
        exceeds_soft_limit=over > envs.USAGE_EXPORT_SOFT_ROWS,
        exceeds_hard_limit=exceeds_hard,
        suggested_max_days=(
            shorten_range_days(request, over, envs.USAGE_EXPORT_MAX_ROWS)
            if exceeds_hard
            else None
        ),
        # The remedies are computed here, not in the client, so the advice
        # before the click matches the advice after a rejection verbatim.
        suggestions=(
            export_suggestions(
                request, over, envs.USAGE_EXPORT_MAX_ROWS, split_parts=split_parts
            )
            if exceeds_hard
            else []
        ),
        split_parts=split_parts,
        # A split export is always CSV (it is the only format that streams),
        # so once splitting is the way out, that is the format the user will
        # get. Saying so here is what keeps the promise before the click equal
        # to the file after it.
        effective_format=(
            USAGE_EXPORT_FORMAT_CSV
            if exceeds_hard
            else effective_export_format(
                request.format, [estimate.total for estimate in estimates]
            )
        ),
    )


def export_too_large(
    request, sheet_key: str, total: int, limit: int, split_parts: Optional[int] = None
) -> InvalidException:
    """The structured over-limit error, identical on every export endpoint.

    The message alone is not actionable — "narrow the range" is true and
    useless. ``details`` carries the numbers plus the moves the UI renders as
    buttons, from the same helper the estimate uses.
    """
    return InvalidException(
        message=(
            f"Result set too large ({total} rows, limit {limit}). "
            "Narrow the date range or split the export."
        ),
        details={
            "kind": "export_too_large",
            "sheet": sheet_key,
            "total": total,
            "limit": limit,
            "suggestions": export_suggestions(
                request, total, limit, split_parts=split_parts
            ),
        },
    )


def split_too_many_parts(planned_members: int) -> InvalidException:
    """The structured refusal when a split would write more files than allowed."""
    return InvalidException(
        message=(
            f"Splitting would produce {planned_members} files, over the "
            f"{envs.USAGE_EXPORT_MAX_SPLIT_MEMBERS} limit. Narrow the date "
            "range or export fewer tables at once."
        ),
        details={
            "kind": "export_split_too_many_parts",
            "total": planned_members,
            "limit": envs.USAGE_EXPORT_MAX_SPLIT_MEMBERS,
        },
    )


def split_member_name(
    sheet_key: str, index: int, parts: int, many: bool, *, prefix: str = "usage"
) -> str:
    """Name for one part of a split export.

    Zero-padded so a plain lexical sort is chronological, and carrying
    ``of-MM`` so a consumer can tell at a glance whether they have the whole
    set. The parts are row slices, not date ranges, so the name deliberately
    does NOT claim a period — the rows inside are date-ordered, and the
    trailer records the row range.

    ``prefix`` names the export the file came from, so a part matches the
    un-split download it corresponds to (``storage_by_volume_<dates>.csv``).
    Multi-sheet archives put each sheet in its own directory instead, where
    the directory already carries that role.
    """
    width = len(str(parts))
    part = f"part-{index:0{width}d}-of-{parts}"
    if many:
        return f"by_{sheet_key}/{part}.csv"
    return f"{prefix}_by_{sheet_key}_{part}.csv"


def export_columns_payload(
    keys: Sequence[str], titles: Sequence[str]
) -> List[UsageExportColumn]:
    """The sheet's columns, keyed and titled, for the preview to render."""
    return [UsageExportColumn(key=key, title=title) for key, title in zip(keys, titles)]
