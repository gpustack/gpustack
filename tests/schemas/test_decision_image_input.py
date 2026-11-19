"""The decision-API image-parameter contract published through model.meta.

vLLM's /v1/systemone schema (``extra="forbid"``) accepts ``images`` only, so
evaluation writes ``meta["decision_image_input"] = "images"`` for a decision
deployment served by vLLM. The Model controller's categories/meta sync carries
the key onto the route row and ``/v1/models?with_meta=true`` hands it to
clients (the Playground's decision dropdown). Engines without a declaration
(the vLLM-SR d3 series accepts both spellings) carry no key: a missing key
means "keep the current request construction".
"""

import pytest

from gpustack.schemas import models


@pytest.mark.parametrize(
    "backend,expected",
    [
        ("vLLM", "images"),
        ("vllm", "images"),
        (" VLLM ", "images"),
        (None, None),
        ("", None),
        ("SGLang", None),
        ("Custom", None),
        ("MindIE", None),
    ],
)
def test_declaration_by_backend(backend, expected):
    assert models.decision_image_input(backend) == expected


def _model(categories=None, backend=None, meta=None):
    return models.Model(
        id=1,
        name="m1",
        source=models.SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/m",
        categories=categories,
        backend=backend,
        meta=meta,
    )


def test_decision_vllm_model_gains_the_declaration():
    model = _model(categories=["decision"], backend="vLLM", meta={"keep": 1})
    assert models.apply_decision_image_input_meta(model) is True
    assert model.meta == {"keep": 1, "decision_image_input": "images"}


def test_undeclared_backends_write_no_key():
    # get_backend resolves a bare backend to vLLM, so only an explicit
    # non-vLLM backend (or a non-decision model) stays keyless.
    for kwargs in (
        dict(categories=["decision"], backend="Custom"),
        dict(categories=["llm"], backend="vLLM"),
        dict(categories=None, backend="vLLM"),
    ):
        model = _model(meta={"keep": 1}, **kwargs)
        assert models.apply_decision_image_input_meta(model) is False
        assert model.meta == {"keep": 1}


def test_idempotent_when_the_key_is_already_current():
    model = _model(categories=["decision"], backend="vLLM")
    assert models.apply_decision_image_input_meta(model) is True
    assert models.apply_decision_image_input_meta(model) is False
    assert model.meta == {"decision_image_input": "images"}


def test_a_stale_declaration_is_cleared_when_it_no_longer_applies():
    # The category or backend changed after the key was written: the key
    # must not outlive the state that justified it.
    model = _model(
        categories=["llm"],
        backend="vLLM",
        meta={"decision_image_input": "images", "keep": 1},
    )
    assert models.apply_decision_image_input_meta(model) is True
    assert model.meta == {"keep": 1}


def test_clearing_is_a_noop_without_the_key():
    model = _model(categories=["llm"], backend="vLLM")
    assert models.apply_decision_image_input_meta(model) is False
    assert not model.meta
