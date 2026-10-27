"""``Model.native_anthropic_api`` has to survive the API round trip.

Not a test of pydantic: the risk is that a write path grows a hand-written
field list -- as ``inference_backend``'s create/update/from-yaml handlers all
have -- and drops the column without an error. These assertions fail the moment
one of the Model handlers stops being ``model_dump``-driven.
"""

from gpustack.schemas.models import (
    Model,
    ModelCreate,
    ModelUpdate,
    SourceEnum,
)
from tests.utils.model import new_model


def _payload(**overrides) -> dict:
    payload = {
        "name": "m",
        "source": SourceEnum.HUGGING_FACE,
        "huggingface_repo_id": "repo/m",
        "replicas": 1,
    }
    payload.update(overrides)
    return payload


def test_create_carries_the_flag():
    created = ModelCreate(**_payload(native_anthropic_api=True))
    # create_model persists ``model_in.model_dump(exclude={"enable_model_route"})``.
    assert created.model_dump()["native_anthropic_api"] is True


def test_it_defaults_to_off():
    """Absent means "translate", which is the pre-existing behavior -- and it is
    a real ``False`` rather than a None, so there is one spelling of "no"."""
    assert ModelCreate(**_payload()).native_anthropic_api is False


def test_update_only_touches_what_was_sent():
    """``ActiveRecordMixin.update`` applies ``model_fields_set``, so a PATCH that
    omits the flag must leave a previously declared value standing rather than
    resetting it to the default."""
    assert "native_anthropic_api" not in ModelUpdate(**_payload()).model_fields_set
    assert (
        "native_anthropic_api"
        in ModelUpdate(**_payload(native_anthropic_api=False)).model_fields_set
    )


def test_a_model_copy_keeps_it():
    """Several controllers duplicate a row with ``Model(**model.model_dump())``
    or ``Model.model_validate(model.model_dump())`` -- a field the dump drops
    would silently reset on every copy."""
    model = new_model(1, "m", huggingface_repo_id="repo/m")
    model.native_anthropic_api = True

    assert model.model_dump()["native_anthropic_api"] is True
    assert Model.model_validate(model.model_dump()).native_anthropic_api is True
