"""Read-time backend merges preserve ORM state and shared catalog semantics."""

import pytest
from sqlalchemy import inspect
from sqlalchemy.orm import Session

from gpustack.schemas.inference_backend import InferenceBackend, VersionConfigDict
from gpustack.utils.inference_backend import merge_backend_definitions


def test_merging_session_rows_does_not_share_the_owner_orm_state():
    platform = InferenceBackend(
        backend_name='router-custom', default_run_command='router', enabled=True
    )
    owner = InferenceBackend(
        id=7,
        backend_name='router-custom',
        owner_principal_id=42,
        default_run_command=None,
    )
    with Session() as session:
        session.add(owner)
        state = inspect(owner)
        history = state.attrs.default_run_command.history
        merged = merge_backend_definitions(platform, owner)
        assert inspect(merged) is not state
        assert inspect(merged).session is None
        assert inspect(merged).transient
        assert merged.default_run_command == 'router'
        assert owner.default_run_command is None
        assert state.attrs.default_run_command.history == history


@pytest.mark.parametrize(
    'platform_enabled,owner_enabled,expected',
    [
        (True, False, True),
        (False, True, True),
        (False, False, False),
        (True, None, True),
    ],
)
def test_catalog_enabled_is_the_union_of_platform_and_org(
    platform_enabled, owner_enabled, expected
):
    platform = InferenceBackend(backend_name='router-custom', enabled=platform_enabled)
    owner = InferenceBackend(
        backend_name='router-custom',
        owner_principal_id=42,
        enabled=owner_enabled,
        version_configs=VersionConfigDict(root={}),
    )
    assert merge_backend_definitions(platform, owner).enabled is expected
