"""The decision-service ping: /v1/systemone through the real decision path.

Unlike test-model's OpenAI chat ping, the decision test POSTs the service's
own model_selection question — the same call the gateway plugin makes per
request — so a pass certifies endpoint, token and the decision path at once.
"""

import json
from typing import Callable, Dict, List

import httpx
import pytest

from gpustack.api.exceptions import InvalidException
from gpustack.routes import model_provider as route_module
from gpustack.schemas.model_provider import (
    ModelProviderCreate,
    ModelProviderTypeEnum,
    OpenAIConfig,
    TypesafeConfig,
    TestDecisionModelInput,
)


def _stub_upstream(monkeypatch, respond: Callable[[httpx.Request], httpx.Response]):
    """Answer the decision test from a stub, recording what was sent."""
    asked: List[Dict] = []
    real_client = httpx.AsyncClient

    def factory(*, base_url, **_kwargs):
        def record(request: httpx.Request) -> httpx.Response:
            asked.append(
                {
                    "base": str(base_url),
                    "path": request.url.path,
                    "json": json.loads(request.content),
                    "auth": request.headers.get("Authorization"),
                }
            )
            return respond(request)

        return real_client(transport=httpx.MockTransport(record), base_url=base_url)

    monkeypatch.setattr(route_module.httpx, "AsyncClient", factory)
    return asked


def _systemone(endpoint="http://jev.tenant-a.internal:8010", model=None):
    return TypesafeConfig.model_validate(
        {
            "type": ModelProviderTypeEnum.GPUSTACK_LB_TYPESAFE.value,
            "endpoint": endpoint,
            **({"model": model} if model else {}),
        }
    )


def _verdict_ok() -> httpx.Response:
    # The official ChoiceAnswer shape: the verdict nests under
    # answers.model_selection with choice/confidence/probabilities all
    # present (api.typesafe.ai/openapi.json).
    return httpx.Response(
        200,
        json={
            "model": "jev-latest",
            "answers": {
                "model_selection": {
                    "type": "choice",
                    "choice": "candidate-a",
                    "confidence": 0.9,
                    "probabilities": {"candidate-a": 0.9, "candidate-b": 0.1},
                }
            },
            "usage": {"input_tokens": 120, "output_tokens": 12},
        },
    )


async def _test_decision(config, model_name=None, api_token="sk-test"):
    return await route_module.try_decision_model_with_provider(
        TestDecisionModelInput(
            api_token=api_token, config=config, model_name=model_name
        )
    )


class TestDecisionPing:
    @pytest.mark.asyncio
    async def test_the_service_decision_endpoint_is_pinged(self, monkeypatch):
        asked = _stub_upstream(monkeypatch, lambda _request: _verdict_ok())

        result = await _test_decision(_systemone())

        assert result.accessible is True
        assert len(asked) == 1
        assert asked[0]["path"] == "/v1/systemone"
        assert asked[0]["base"] == "http://jev.tenant-a.internal:8010"
        assert asked[0]["auth"] == "Bearer sk-test"
        # The jevcompat wire contract: state + the choice question under
        # questions.model_selection, with >= 2 criteria so the service
        # exercises the same path a route's request takes.
        body = asked[0]["json"]
        assert body["state"]
        question = body["questions"]["model_selection"]
        assert question["type"] == "choice"
        assert len(question["criteria"]) >= 2
        assert "model" not in body  # no alias known: omitted, not guessed

    @pytest.mark.asyncio
    async def test_hosted_default_endpoint_is_used(self, monkeypatch):
        asked = _stub_upstream(monkeypatch, lambda _request: _verdict_ok())

        # no custom base url: the TypeSafe hosted default is the target
        await _test_decision(_systemone(endpoint=None))

        assert asked[0]["base"] == "https://api.typesafe.ai"

    @pytest.mark.asyncio
    async def test_model_name_defaults_to_the_provider_model(self, monkeypatch):
        asked = _stub_upstream(monkeypatch, lambda _request: _verdict_ok())

        result = await _test_decision(_systemone(model="jev-preview"))

        assert result.model_name == "jev-preview"
        # the alias rides the request as the jevcompat ``model`` field
        assert asked[0]["json"]["model"] == "jev-preview"

    @pytest.mark.asyncio
    async def test_model_name_override_wins(self, monkeypatch):
        asked = _stub_upstream(monkeypatch, lambda _request: _verdict_ok())

        result = await _test_decision(
            _systemone(model="jev-preview"), model_name="jev-latest"
        )

        assert result.model_name == "jev-latest"
        assert asked[0]["json"]["model"] == "jev-latest"

    @pytest.mark.asyncio
    async def test_non_2xx_reports_inaccessible_as_is(self, monkeypatch):
        _stub_upstream(
            monkeypatch,
            lambda _request: httpx.Response(
                401, json={"error": {"message": "invalid api key"}}
            ),
        )

        result = await _test_decision(_systemone())

        assert result.accessible is False
        assert "401" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_tokenless_ping_omits_the_header(self, monkeypatch):
        # A self-hosted service may not require auth; on the /{id} path a
        # provider without api_tokens pings with no Authorization header
        # rather than the test being refused.
        asked = _stub_upstream(monkeypatch, lambda _request: _verdict_ok())

        result = await route_module._try_decision_model(_systemone(), None, None, None)

        assert result.accessible is True
        assert asked[0]["auth"] is None


class TestDecisionPingRejections:
    @pytest.mark.asyncio
    async def test_html_page_with_200_is_not_a_pass(self, monkeypatch):
        # Reachable is not usable: a proxy's HTML page answering 200 must
        # not report the decision path as working.
        _stub_upstream(
            monkeypatch,
            lambda _request: httpx.Response(
                200, text="<html><body>502 Bad Gateway</body></html>"
            ),
        )

        result = await _test_decision(_systemone())

        assert result.accessible is False
        assert "non-JSON" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_json_error_envelope_with_200_is_not_a_pass(self, monkeypatch):
        _stub_upstream(
            monkeypatch,
            lambda _request: httpx.Response(200, json={"error": {"message": "boom"}}),
        )

        result = await _test_decision(_systemone())

        assert result.accessible is False
        assert "error envelope" in (result.error_message or "")

    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"status": "ok"},
            {"models": []},
            {"answers": {}},
            # choice at the top level (the pre-contract guess) is NOT a
            # valid verdict: the answer nests under answers.model_selection
            {"choice": "candidate-a"},
            # answer missing required confidence/probabilities
            {
                "answers": {
                    "model_selection": {
                        "type": "choice",
                        "choice": "candidate-a",
                    }
                }
            },
        ],
    )
    @pytest.mark.asyncio
    async def test_unusable_verdicts_are_not_a_pass(self, monkeypatch, body):
        # Reachable is not usable: anything the gateway cannot build a rank
        # entry from (no non-blank choice) reports failure with the body.
        _stub_upstream(monkeypatch, lambda _request: httpx.Response(200, json=body))

        result = await _test_decision(_systemone())

        assert result.accessible is False

    @pytest.mark.asyncio
    async def test_non_decision_config_is_rejected(self):
        with pytest.raises(InvalidException) as exc_info:
            await _test_decision(OpenAIConfig(type=ModelProviderTypeEnum.OPENAI))
        assert "not a decision service" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_adhoc_ping_without_a_token_sends_no_header(self, monkeypatch):
        # A self-hosted decision service may run without auth: an omitted
        # (or empty) token is valid on the ad-hoc route too, and the ping
        # simply carries no Authorization header. The hosted default
        # rejects that with 401, reported as-is.
        asked = _stub_upstream(monkeypatch, lambda _request: _verdict_ok())

        result = await route_module.try_decision_model_with_provider(
            TestDecisionModelInput(api_token=None, config=_systemone())
        )

        assert result.accessible is True
        assert asked[0]["auth"] is None


class TestFailoverTokenValidation:
    """The multi-token failover prerequisite is an ai-proxy concern: decision
    services fail their keys over inside the wasm plugin, not through
    ai-proxy's failover config, so they are exempt from the llm-model
    requirement."""

    def test_decision_provider_allows_multiple_tokens_without_llm_model(self):
        provider = ModelProviderCreate.model_validate(
            {
                "name": "jev",
                "config": {"type": ModelProviderTypeEnum.GPUSTACK_LB_TYPESAFE.value},
                "models": [{"name": "jev-latest", "category": "decision"}],
                "api_tokens": [{"input": "primary-key"}, {"input": "fallback-key"}],
            }
        )
        route_module.validate_provider(provider)

    def test_inference_provider_still_requires_llm_model_for_failover(self):
        provider = ModelProviderCreate.model_validate(
            {
                "name": "openai",
                "config": {"type": ModelProviderTypeEnum.OPENAI.value},
                "models": [{"name": "text-embedding-x", "category": "embedding"}],
                "api_tokens": [{"input": "primary-key"}, {"input": "fallback-key"}],
            }
        )
        with pytest.raises(InvalidException) as exc_info:
            route_module.validate_provider(provider)
        assert "llm model is required" in exc_info.value.message


class TestTypeTransitionGuard:
    @pytest.mark.asyncio
    async def test_type_flip_to_decision_rejected_while_targets_exist(
        self, monkeypatch
    ):
        # An ordinary provider already serving inference targets cannot be
        # flipped to gpustack-lb-typesafe: the targets survive (their
        # validation only runs on target writes) while the provider leaves
        # the ai-proxy catalogue -- routes pointing at nothing.
        from types import SimpleNamespace

        from gpustack.api.exceptions import InvalidException
        from gpustack.schemas.model_provider import (
            ModelProvider,
            ModelProviderTypeEnum,
            ModelProviderUpdate,
            OpenAIConfig,
            ProviderModel,
        )
        import gpustack.routes.model_provider as route_module
        import gpustack.schemas.model_routes as mrt

        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        provider = ModelProvider(
            id=5,
            name="openai",
            config=OpenAIConfig(type=ModelProviderTypeEnum.OPENAI),
            models=[ProviderModel(name="gpt-x")],
            created_at=now,
            updated_at=now,
            owner_principal_id=1,
        )
        provider.api_tokens = ["secret-token"]

        async def one_by_id(cls, session, id, **kw):
            return provider

        async def all_by_field(cls, session, field, value):
            assert (field, value) == ("provider_id", 5)
            return [SimpleNamespace(id=1)]

        monkeypatch.setattr(ModelProvider, "one_by_id", classmethod(one_by_id))
        monkeypatch.setattr(
            mrt.ModelRouteTarget, "all_by_field", classmethod(all_by_field)
        )
        monkeypatch.setattr(
            route_module, "assert_resource_visible", lambda *a, **k: None
        )
        update = ModelProviderUpdate.model_validate(
            {
                "name": "openai",
                "config": {"type": ModelProviderTypeEnum.GPUSTACK_LB_TYPESAFE.value},
                "models": [{"name": "gpt-x"}],
            }
        )
        with pytest.raises(InvalidException) as exc_info:
            await route_module.update_model_provider(
                session=object(), ctx=SimpleNamespace(), id=5, input=update
            )
        assert "cannot change to a decision-service type" in exc_info.value.message

        # the same provider with no targets flips freely
        async def no_targets(cls, session, field, value):
            return []

        monkeypatch.setattr(
            mrt.ModelRouteTarget, "all_by_field", classmethod(no_targets)
        )
        from types import SimpleNamespace  # noqa: F401 (used by helper)

        updated = await _update_without_side_effects(monkeypatch, provider, update)
        assert updated is not None


async def _update_without_side_effects(monkeypatch, provider, update):
    from types import SimpleNamespace

    import gpustack.routes.model_provider as route_module
    from gpustack.schemas.model_provider import ModelProvider

    async def do_update(self, source=None, **kw):
        return provider

    monkeypatch.setattr(ModelProvider, "update", do_update)
    return await route_module.update_model_provider(
        session=object(),
        ctx=SimpleNamespace(),
        id=provider.id,
        input=update,
    )


@pytest.mark.asyncio
async def test_official_typesafe_response_shape_is_a_pass(monkeypatch):
    # Reproduced from api.typesafe.ai/openapi.json (PR review): the verdict
    # nests under answers.model_selection; the earlier top-level choice
    # check rejected this exact body.
    body = {
        "model": "jev-latest",
        "answers": {
            "model_selection": {
                "type": "choice",
                "choice": "candidate-a",
                "confidence": 0.9,
                "probabilities": {"candidate-a": 0.9, "candidate-b": 0.1},
            }
        },
        "usage": {"input_tokens": 120, "output_tokens": 12},
    }
    _stub_upstream(monkeypatch, lambda _request: httpx.Response(200, json=body))

    result = await _test_decision(_systemone())

    assert result.accessible is True
