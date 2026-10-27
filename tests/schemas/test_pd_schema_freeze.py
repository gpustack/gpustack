"""The frozen PD schema, and the one discipline it turns on.

The status fields must not be writable by a client. They are declared on the
`Model` table and on `ModelPublic`, and deliberately NOT on `ModelBase` —
because `ModelUpdate` inherits `ModelBase`, and the UI issues whole-object
PUTs (start/stop, inline replica edits), so anything reachable from
`ModelBase` gets written back by the client. `ready_replicas` sits on
`ModelSpecBase` for historical reasons and the frontend has to strip it by
hand; this test is what stops that list from growing.
"""

import sqlalchemy as sa

from gpustack.schemas.models import (
    Model,
    ModelCreate,
    ModelInstance,
    ModelPublic,
    ModelStateEnum,
    ModelUpdate,
)

# Written only by `sync_model_status`, never accepted from a request.
SERVER_OWNED = ("state", "state_message", "role_status", "stale", "degradations")

# User intent: these ARE part of the request body.
SPEC_FIELDS = ("roles", "disaggregation")


def test_server_owned_status_is_a_column():
    columns = {c.name for c in Model.__table__.columns}
    for name in SERVER_OWNED:
        assert name in columns, name


def test_server_owned_status_is_not_client_writable():
    for name in SERVER_OWNED:
        assert name not in ModelUpdate.model_fields, name
        assert name not in ModelCreate.model_fields, name


def test_server_owned_status_is_readable():
    for name in SERVER_OWNED:
        assert name in ModelPublic.model_fields, name


def test_spec_fields_are_client_writable_and_readable():
    for name in SPEC_FIELDS:
        assert name in ModelUpdate.model_fields, name
        assert name in ModelCreate.model_fields, name
        assert name in ModelPublic.model_fields, name


def test_the_status_columns_are_strings_not_pg_enums():
    """A bare `Optional[SomeEnum]` annotation maps to `sa.Enum`, which breaks
    twice over: asyncpg then casts every read and write to a Postgres type the
    migration never created, and sa.Enum persists member *names* so the row
    would hold "RUNNING" while the API, the enum's own value and the `?state=`
    filter all say "running". Following `CacheService.state`, these are plain
    strings."""
    state = Model.__table__.c.state.type
    assert not isinstance(state, sa.Enum), repr(state)
    assert isinstance(state, sa.String)


def test_the_enum_serialises_as_its_value():
    assert ModelStateEnum.RUNNING.value == "running"
    assert str(ModelStateEnum.RUNNING) == "running"


def test_instance_group_columns_exist_and_group_id_is_indexed():
    columns = {c.name for c in ModelInstance.__table__.columns}
    for name in ("role", "group_id", "spec_digest", "named_ports"):
        assert name in columns, name
    # The reconcilers group by it.
    assert any(
        "group_id" in {c.name for c in index.columns}
        for index in ModelInstance.__table__.indexes
    )
