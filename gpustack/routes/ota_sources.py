"""Per-kind source configuration for the three kinds of OTA content.

A top-level resource rather than a ``/source`` child of each consumer. Hanging it
off ``/inference-backends`` put a fixed segment inside the ``/{id}`` namespace,
where it resolved only because it was mounted first — a reader cannot tell
``source`` from a backend id, and moving the mount would silently turn the
endpoint into a 422. One family also matches how the content is administered:
all three kinds are configured on one screen, beside ``/source-probe``.

``kind`` carries the same names ``GET /source-probe`` reports its ``kinds`` map
under, so a client that read the status addresses the configuration without a
translation table.

More sources per kind grow *below* ``{kind}`` (``/{kind}/sources[/{id}]``): a
source id is per-table, so ``{kind}`` never gives up its segment to an id.
"""

from enum import Enum
from typing import Dict

from fastapi import APIRouter, Request

from gpustack.routes import inference_backend, model_sets
from gpustack.server.deps import SessionDep
from gpustack.server.sources.probe import running_refresher
from gpustack.server.sources.routes import (
    SourceConfig,
    SourceConfigSpec,
    SourceConfigUpsert,
    SourceWriteResult,
    delete_source_config,
    get_source_config,
    reload_source_config,
    update_source_config,
)

router = APIRouter()


class SourceKind(str, Enum):
    """Which kind of content a request configures. An enum, so an unknown kind is
    a 422 naming the three rather than a 404 from a dict lookup, and a generated
    client gets the union. Values are ``OfficialKind.name`` verbatim.
    """

    CATALOG = "catalog"
    COMMUNITY_BACKEND = "community-backend"
    BUILT_IN_BACKEND = "built-in-backend"


# Each kind's binding for the shared engine, from the module that owns that
# content — the two backend specs carry in-use checks that live there.
_SPECS: Dict[SourceKind, SourceConfigSpec] = {
    SourceKind.CATALOG: model_sets.CATALOG_SOURCE_SPEC,
    SourceKind.COMMUNITY_BACKEND: inference_backend.COMMUNITY_BACKEND_SPEC,
    SourceKind.BUILT_IN_BACKEND: inference_backend.BUILTIN_BACKEND_SPEC,
}


@router.get("/{kind}")
async def get_config(kind: SourceKind, session: SessionDep) -> SourceConfig:
    return await get_source_config(session, _SPECS[kind])


@router.put("/{kind}")
async def update_config(
    kind: SourceKind, session: SessionDep, source_in: SourceConfigUpsert
) -> SourceWriteResult:
    return await update_source_config(session, _SPECS[kind], source_in)


@router.delete("/{kind}")
async def delete_config(kind: SourceKind, session: SessionDep) -> SourceConfig:
    return await delete_source_config(session, _SPECS[kind])


@router.post("/{kind}/reload")
async def reload_config(
    kind: SourceKind, request: Request, session: SessionDep
) -> SourceWriteResult:
    return await reload_source_config(
        session, _SPECS[kind], running_refresher(request.app.state)
    )
