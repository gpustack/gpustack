"""Packaging a sized export into an HTTP response.

The token and resource exports differ in how their rows are produced and in
nothing else that a downloader can see: the same formats, the same file names,
the same trailer, the same archive layout. So a route resolves its sheets into
:class:`ExportSheetPlan` values — a name, a header row, a row count and a way
to open the row stream — and the two functions here decide what file that
becomes.

Keeping it in one place is what stops the two tabs' downloads from drifting
apart in shape while claiming to be the same feature.
"""

import contextlib
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, List, Optional, Sequence

from fastapi import Response
from fastapi.responses import StreamingResponse

from gpustack import envs
from gpustack.schemas.usage import USAGE_EXPORT_FORMAT_XLSX
from gpustack.utils.export_limits import (
    attachment_headers,
    export_split_plan,
    split_member_name,
    split_too_many_parts,
)
from gpustack.utils.tabular_export import (
    build_xlsx,
    stream_csv,
    stream_zip,
    take_rows,
)

XLSX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


@dataclass
class ExportSheetPlan:
    """One logical table, resolved to everything the writer needs.

    ``rows`` is a factory rather than an iterator because a split export opens
    exactly one row stream per sheet and slices it, while the other formats
    open one per file; the plan stays the same either way.
    """

    key: str
    name: Optional[str]
    columns: List[str]
    total: int
    rows: Callable[[], AsyncIterator[Sequence[Any]]]


def trailer_context(effective_scope: str, organization_id: Optional[int]) -> str:
    """What every file's trailer says about the query behind it.

    The scope is the EFFECTIVE one, not the requested one, so a file exported
    by a user whose ``all`` was answered with their own rows says so — the
    reader of the file cannot otherwise tell.
    """
    return f"scope={effective_scope} organization={organization_id or '*'}"


def _sheet_trailer(plan: ExportSheetPlan, context: str) -> str:
    return f"rows={plan.total} {context}"


async def export_response(
    plans: List[ExportSheetPlan],
    *,
    request,
    prefix: str,
    export_format: str,
    context: str,
) -> Response:
    """One sheet becomes a file, several become an archive or a workbook."""
    stamp = f"{request.start_date}_{request.end_date}"

    if export_format == USAGE_EXPORT_FORMAT_XLSX:
        payload = await build_xlsx(
            (plan.name or plan.key, plan.columns, plan.rows()) for plan in plans
        )
        return Response(
            content=payload,
            media_type=XLSX_MEDIA_TYPE,
            headers=attachment_headers(f"{prefix}_{stamp}.xlsx"),
        )

    if len(plans) == 1:
        plan = plans[0]
        return StreamingResponse(
            stream_csv(
                plan.columns,
                plan.rows(),
                trailer=_sheet_trailer(plan, context),
            ),
            media_type="text/csv; charset=utf-8",
            headers=attachment_headers(f"{prefix}_by_{plan.key}_{stamp}.csv"),
        )

    members = [
        (
            f"by_{plan.key}.csv",
            stream_csv(
                plan.columns,
                plan.rows(),
                trailer=_sheet_trailer(plan, context),
            ),
        )
        for plan in plans
    ]
    return StreamingResponse(
        stream_zip(members),
        media_type="application/zip",
        headers=attachment_headers(f"{prefix}_{stamp}.zip"),
    )


def split_export_response(
    plans: List[ExportSheetPlan],
    *,
    request,
    prefix: str,
    context: str,
    limit: int,
) -> StreamingResponse:
    """Deliver an over-large export as one archive of row-sliced files.

    This is what keeps the hard limit from being a dead end: without it the
    only answer to "my range is too big" is "make it smaller".

    **Parts are slices of the ROW STREAM, not narrower date ranges.** The
    aggregation has already happened by the time these rows exist, so a cut
    between two of them re-aggregates nothing — which is why every grouping
    can be split, why the part count is exactly ``ceil(total / limit)``, and
    why one cursor can feed them all (a row inserted mid-export cannot land in
    two files or none).

    Always CSV, whatever format was requested. Not a format constraint — every
    part fits a worksheet — but a throughput one: xlsx must be assembled in
    full before any of it is valid, and a split multiplies that by the part
    count. The estimate reports ``effective_format``, so this is announced
    rather than sprung.
    """
    parts_by_sheet = export_split_plan({plan.key: plan.total for plan in plans}, limit)
    planned_members = sum(parts_by_sheet.values())
    if planned_members > envs.USAGE_EXPORT_MAX_SPLIT_MEMBERS:
        raise split_too_many_parts(planned_members)

    many = len(plans) > 1

    async def members():
        for plan in plans:
            parts = parts_by_sheet[plan.key]
            # ONE row stream per sheet; each part takes the next slice off it.
            # ``aclosing`` releases it — the cursor, plus the enrichment
            # session behind it — once the last part is written or the client
            # walks away. A part that ends exactly on the end of the data
            # never reaches the source's own exit, so nothing else would.
            rows = plan.rows()
            async with contextlib.aclosing(rows):
                for index in range(1, parts + 1):
                    first = (index - 1) * limit + 1
                    last = min(index * limit, plan.total)
                    yield (
                        split_member_name(plan.key, index, parts, many, prefix=prefix),
                        stream_csv(
                            plan.columns,
                            take_rows(rows, limit),
                            trailer=(
                                f"rows={last - first + 1} part={index}/{parts} "
                                f"range={first}-{last} of {plan.total} {context}"
                            ),
                        ),
                    )

    stamp = f"{request.start_date}_{request.end_date}"
    return StreamingResponse(
        stream_zip(members()),
        media_type="application/zip",
        headers=attachment_headers(f"{prefix}_{stamp}_split.zip"),
    )
