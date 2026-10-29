import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from gpustack.schemas.models import (
    CategoryEnum,
    Model,
    ModelCreate,
    ModelUpdate,
    is_decision_model,
)
from gpustack.schemas.model_routes import ModelRouteUpdate


def test_decision_category_in_enum():
    assert CategoryEnum.DECISION.value == "decision"


def test_route_accepts_decision_category():
    route = ModelRouteUpdate(name="d3-route", categories=["decision"])
    assert route.categories == ["decision"]


def test_route_still_rejects_unknown_category():
    with pytest.raises(ValidationError):
        ModelRouteUpdate(name="bad-route", categories=["vqa"])


def test_is_decision_model():
    assert is_decision_model(Model(id=1, name="d3", categories=["decision"]))
    assert not is_decision_model(Model(id=2, name="qwen", categories=["llm"]))
    assert not is_decision_model(Model(id=3, name="plain"))
    # categories is nullable: the validator passes None through untouched,
    # so membership must not assume a list.
    assert not is_decision_model(Model(id=4, name="bare", categories=None))


def _model_payload(**overrides):
    payload = {
        "name": "d3",
        "source": "huggingface",
        "huggingface_repo_id": "vllm-sr/d3-flash",
    }
    payload.update(overrides)
    return payload


def test_model_write_rejects_unknown_category():
    # The write boundary is ModelCreate/ModelUpdate (non-table SQLModel —
    # table classes skip validators on construction). A typo'd decision
    # category would silently create a model that is neither a decision
    # provider nor filterable as one: the identity IS the string, and
    # nothing auto-corrects it.
    with pytest.raises(ValidationError, match="Invalid category"):
        ModelCreate.model_validate(_model_payload(categories=["Decision"]))


def test_model_update_round_trips_stored_categories():
    # ModelUpdate is also the round-trip vessel for internal whole-object
    # PUTs (workers persisting backend_version via ModelUpdate(**dump), the
    # UI echoing the row): the schema must construct from a legacy row's
    # categories without raising, or those updates would brick on it. The
    # update-side whitelist lives in the route layer, against the stored
    # row (see tests/routes/test_model_validation.py).
    update = ModelUpdate.model_validate(_model_payload(categories=["vqa"]))
    assert update.categories == ["vqa"]


def test_model_write_accepts_every_enum_category():
    for value in CategoryEnum:
        payload = ModelCreate.model_validate(_model_payload(categories=[value]))
        assert value in payload.categories


def test_stored_unknown_category_still_validates_on_the_read_path():
    # The whitelist lives on the write schemas only: read-side validation
    # (the watch stream's ModelPublic conversion, the controllers'
    # Model.model_validate copies, the RoleEffectiveModel projections) runs
    # over stored rows, and a legacy row holding a category outside the
    # enum — the column predates the whitelist — must keep loading.
    from gpustack.schemas.models import ModelPublic
    from tests.utils.model import new_model

    # Table classes skip validators on construction, which is how a legacy
    # row comes back from the ORM.
    stored = new_model(1, "legacy", huggingface_repo_id="repo/m", categories=["vqa"])
    assert stored.categories == ["vqa"]
    # The controllers' row-copy form (see the ai-proxy / replica-sync
    # notify paths) re-validates the full dump.
    assert Model.model_validate(stored.model_dump()).categories == ["vqa"]
    now = datetime.now(timezone.utc)
    public = ModelPublic.model_validate(
        {
            **stored.model_dump(),
            "id": 7,
            "created_at": now,
            "updated_at": now,
        }
    )
    assert public.categories == ["vqa"]
