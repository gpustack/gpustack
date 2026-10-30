"""Who a benchmark measures, and where its container runs.

Two separate facts. Under PD a group answers only through its router, so the
client cannot name the member being measured; and the router holds no weights,
so its worker is not a machine the tokenizer can be read from. The target and
the placement are therefore resolved independently.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from starlette.requests import Request

from gpustack.api.exceptions import BadRequestException, NotFoundException
from gpustack.api.tenant import TenantContext
from gpustack.routes import benchmarks as route
from gpustack.routes import clusters as cluster_routes
from gpustack.schemas.benchmark import (
    BenchmarkCreate,
    BenchmarkSnapshot,
    BenchmarkTargetModeEnum,
)
from gpustack.schemas.model_routes import TargetStateEnum
from gpustack.schemas.models import (
    DisaggregationSpec,
    Model,
    ModelConfigSnapshot,
    ModelInstance,
    ModelInstanceStateEnum,
    PDModeEnum,
    RoleNameEnum,
    RoleSpec,
    SourceEnum,
)


def _group_model(**kwargs) -> Model:
    return Model(
        id=kwargs.pop("id", 1),
        name=kwargs.pop("name", "g"),
        cluster_id=1,
        replicas=1,
        # A column the DB requires, so every Model these tests stand in for has
        # one. `Model` is a SQLModel table and skips validation on construction,
        # which is what let it be omitted -- until the snapshot path started
        # projecting roles through `RoleEffectiveModel.model_validate`, which
        # does validate.
        source=SourceEnum.HUGGING_FACE.value,
        huggingface_repo_id="org/g",
        roles=[
            RoleSpec(name=RoleNameEnum.PREFILL.value, replicas=1),
            RoleSpec(name=RoleNameEnum.DECODE.value, replicas=1),
            RoleSpec(name=RoleNameEnum.ROUTER.value, replicas=1),
        ],
        disaggregation=DisaggregationSpec(mode=PDModeEnum.VLLM_NIXL),
        **kwargs,
    )


def _plain_model(**kwargs) -> Model:
    return Model(
        id=kwargs.pop("id", 1),
        name=kwargs.pop("name", "m"),
        cluster_id=1,
        replicas=1,
        source=SourceEnum.HUGGING_FACE.value,
        huggingface_repo_id="org/m",
        **kwargs,
    )


def _member(name, role=None, id=1, worker_id=10, path="/w", **kwargs) -> ModelInstance:
    return ModelInstance(
        id=id,
        name=name,
        model_id=1,
        model_name="g",
        role=role,
        group_id=kwargs.pop("group_id", "grp" if role else None),
        worker_id=worker_id,
        resolved_path=path,
        state=kwargs.pop("state", ModelInstanceStateEnum.RUNNING),
        model_config_snapshot=kwargs.pop(
            "model_config_snapshot", ModelConfigSnapshot()
        ),
        **kwargs,
    )


def _group_members():
    return [
        _member("p", RoleNameEnum.PREFILL.value, id=1, worker_id=11),
        _member("d", RoleNameEnum.DECODE.value, id=2, worker_id=12),
        # The router holds nothing: no weights on its worker is the whole point.
        _member("r", RoleNameEnum.ROUTER.value, id=3, worker_id=13, path=None),
    ]


def _with_members(members):
    return patch.object(ModelInstance, "all_by_field", AsyncMock(return_value=members))


class TestClusterGatewaySnapshot:
    @pytest.fixture
    def setup(self, monkeypatch):
        model = _plain_model(owner_principal_id=10)
        endpoint = _member("router", worker_id=13, cluster_id=1, owner_principal_id=10)
        placement = _member("decode", worker_id=12, cluster_id=1)
        worker = SimpleNamespace(id=12, cluster_id=7)
        cluster = SimpleNamespace(
            id=7,
            owner_principal_id=10,
            deleted_at=None,
            server_url="https://cluster-gateway.example:8443/prefix/",
        )
        for name, value in (
            ("_resolve_target_model", model),
            ("_resolve_target_endpoint", endpoint),
            ("_resolve_placement", placement),
            ("_resolve_route_name", "qwen"),
            ("get_benchmark_snapshot", BenchmarkSnapshot(route_name="qwen")),
        ):
            monkeypatch.setattr(route, name, AsyncMock(return_value=value))
        worker_lookup = AsyncMock(return_value=worker)
        cluster_lookup = AsyncMock(return_value=cluster)
        monkeypatch.setattr(
            route,
            "WorkerService",
            lambda _: SimpleNamespace(get_by_id=worker_lookup),
        )
        monkeypatch.setattr(route.Cluster, "one_by_id", cluster_lookup)
        monkeypatch.setattr(
            route.Benchmark, "one_by_field", AsyncMock(return_value=None)
        )
        save = AsyncMock(side_effect=lambda session, value: value)
        monkeypatch.setattr(route.Benchmark, "create", save)
        config = SimpleNamespace(
            server_external_url="https://public-gateway.example/",
            trusted_hosts=["request-gateway.example:8443"],
        )
        monkeypatch.setattr(cluster_routes, "get_global_config", lambda: config)
        monkeypatch.setattr(route, "get_global_config", lambda: config)
        return SimpleNamespace(
            model=model,
            worker=worker,
            cluster=cluster,
            worker_lookup=worker_lookup,
            cluster_lookup=cluster_lookup,
            save=save,
            config=config,
            request=Request(
                {
                    "type": "http",
                    "scheme": "https",
                    "path": "/v2/benchmarks",
                    "query_string": b"",
                    "headers": [(b"host", b"request-gateway.example:8443")],
                }
            ),
            ctx=TenantContext(
                user=None,
                is_platform_admin=False,
                current_principal_id=10,
                org_role=None,
            ),
        )

    async def _create(self, setup, mode=BenchmarkTargetModeEnum.ROUTE):
        return await route.create_benchmark(
            None,
            setup.ctx,
            BenchmarkCreate(
                name="bm",
                model_id=1,
                cluster_id=999,
                target_mode=mode,
                dataset_name="Random",
                dataset_input_tokens=128,
                dataset_output_tokens=64,
            ),
            request=setup.request,
        )

    @pytest.mark.asyncio
    async def test_creation_freezes_the_placement_workers_cluster_url(self, setup):
        benchmark = await self._create(setup)

        assert (
            getattr(benchmark.snapshot, "gateway_url", None)
            == "https://cluster-gateway.example:8443/prefix"
        )
        setup.worker_lookup.assert_awaited_once_with(12)
        setup.cluster_lookup.assert_awaited_once_with(None, 7)
        assert benchmark.worker_id == 12
        assert benchmark.cluster_id == 1
        transferred = BenchmarkSnapshot.model_validate_json(
            benchmark.snapshot.model_dump_json()
        )
        assert transferred.gateway_url == benchmark.snapshot.gateway_url
        setup.cluster.server_url = "https://changed.example"
        assert (
            benchmark.snapshot.gateway_url
            == "https://cluster-gateway.example:8443/prefix"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("url", [None, "", "   "])
    async def test_missing_cluster_url_uses_server_external_url(self, setup, url):
        setup.cluster.server_url = url
        setup.config.server_external_url = "https://public-gateway.example:443/prefix/"
        benchmark = await self._create(setup)
        assert (
            benchmark.snapshot.gateway_url
            == "https://public-gateway.example:443/prefix"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("external_url", [None, ""])
    async def test_missing_configured_urls_use_request_entrance(
        self, setup, external_url
    ):
        setup.cluster.server_url = None
        setup.config.server_external_url = external_url
        benchmark = await self._create(setup)
        assert benchmark.snapshot.gateway_url == "https://request-gateway.example:8443"

    @pytest.mark.asyncio
    async def test_request_entrance_preserves_ipv6_and_port(self, setup):
        setup.cluster.server_url = None
        setup.config.server_external_url = None
        setup.config.trusted_hosts = ["[2001:db8::1]:8080"]
        setup.request = Request(
            {
                "type": "http",
                "scheme": "http",
                "path": "/v2/benchmarks",
                "query_string": b"",
                "headers": [(b"host", b"[2001:db8::1]:8080")],
            }
        )
        benchmark = await self._create(setup)
        assert benchmark.snapshot.gateway_url == "http://[2001:db8::1]:8080"

    @pytest.mark.asyncio
    async def test_request_fallback_does_not_trust_raw_forwarding_headers(self, setup):
        setup.cluster.server_url = None
        setup.config.server_external_url = None
        setup.request.scope["headers"].append(
            (b"x-forwarded-host", b"untrusted.example")
        )
        benchmark = await self._create(setup)
        assert benchmark.snapshot.gateway_url == "https://request-gateway.example:8443"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "trusted_hosts",
        [
            None,
            [],
            ["*"],
            ["gateway.example"],
            ["*.example"],
            ["request-gateway.example"],
            ["request-gateway.example:443"],
        ],
    )
    async def test_request_target_must_match_explicit_host_and_port(
        self, setup, trusted_hosts
    ):
        setup.cluster.server_url = None
        setup.config.server_external_url = None
        setup.config.trusted_hosts = trusted_hosts
        with pytest.raises(BadRequestException) as excinfo:
            await self._create(setup)
        assert "trusted_hosts" in excinfo.value.message
        setup.save.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_shared_cluster_owner_cannot_send_worker_token_to_arbitrary_host(
        self, setup
    ):
        setup.cluster.owner_principal_id = 20
        setup.ctx.accessible_cluster_ids = {7}
        setup.cluster.server_url = None
        setup.config.server_external_url = None
        setup.config.trusted_hosts = ["gateway.example"]
        setup.request.scope["headers"] = [(b"host", b"attacker.example")]
        with pytest.raises(BadRequestException):
            await self._create(setup)
        setup.save.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cluster_override", [False, True])
    async def test_configured_entrance_ignores_untrusted_request_host(
        self, setup, cluster_override
    ):
        setup.config.trusted_hosts = None
        setup.request.scope["headers"] = [(b"host", b"attacker.example")]
        if not cluster_override:
            setup.cluster.server_url = None
        benchmark = await self._create(setup)
        expected = (
            setup.cluster.server_url
            if cluster_override
            else setup.config.server_external_url
        )
        assert benchmark.snapshot.gateway_url == expected.rstrip("/")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "scheme,host", [("http", "gateway.example"), ("https", "gateway.example:443")]
    )
    async def test_exact_trusted_host_allows_standard_port(self, setup, scheme, host):
        setup.cluster.server_url = None
        setup.config.server_external_url = None
        setup.config.trusted_hosts = ["GATEWAY.EXAMPLE"]
        setup.request = Request(
            {
                "type": "http",
                "scheme": scheme,
                "path": "/v2/benchmarks",
                "query_string": b"",
                "headers": [(b"host", host.encode())],
            }
        )
        benchmark = await self._create(setup)
        assert benchmark.snapshot.gateway_url == f"{scheme}://{host}"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "host",
        ["gateway.example:bad", "gateway.example:65536", "user@gateway.example:8443"],
    )
    async def test_malformed_request_authority_is_refused(self, setup, host):
        setup.cluster.server_url = None
        setup.config.server_external_url = None
        setup.config.trusted_hosts = ["gateway.example:8443"]
        setup.request.scope["headers"] = [(b"host", host.encode())]
        with pytest.raises(BadRequestException):
            await self._create(setup)
        setup.save.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_other_tenants_model_is_refused_before_looking_up_gateway(
        self, setup
    ):
        setup.model.owner_principal_id = 20
        with pytest.raises(NotFoundException):
            await self._create(setup)
        setup.cluster_lookup.assert_not_awaited()
        setup.save.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_placement_worker_refuses_creation(self, setup):
        setup.worker_lookup.return_value = None
        with pytest.raises(BadRequestException) as excinfo:
            await self._create(setup)
        assert "placement worker" in excinfo.value.message
        setup.save.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("deleted", [False, True])
    async def test_missing_or_deleted_cluster_refuses_creation(self, setup, deleted):
        if deleted:
            setup.cluster.deleted_at = "2026-01-01"
        else:
            setup.cluster_lookup.return_value = None
        with pytest.raises(NotFoundException):
            await self._create(setup)
        setup.save.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_other_tenants_cluster_is_not_exposed(self, setup):
        setup.cluster.owner_principal_id = 20
        with pytest.raises(NotFoundException):
            await self._create(setup)
        setup.save.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_shared_cluster_is_visible(self, setup):
        setup.cluster.owner_principal_id = 20
        setup.ctx.accessible_cluster_ids = {7}
        benchmark = await self._create(setup)
        assert (
            getattr(benchmark.snapshot, "gateway_url", None)
            == "https://cluster-gateway.example:8443/prefix"
        )

    @pytest.mark.asyncio
    async def test_instance_mode_does_not_require_a_gateway(self, setup):
        setup.cluster.server_url = None
        benchmark = await self._create(setup, BenchmarkTargetModeEnum.INSTANCE)
        assert getattr(benchmark.snapshot, "gateway_url", None) is None
        setup.cluster_lookup.assert_not_awaited()


class TestEndpoint:
    """`servable_instances()` decides, which makes this its third caller after
    the gateway's upstream registration and the direct proxy."""

    @pytest.mark.asyncio
    async def test_a_group_resolves_to_its_router(self):
        model = _group_model()
        with _with_members(_group_members()):
            endpoint = await route._resolve_target_endpoint(None, model, None)
        assert endpoint.name == "r"

    @pytest.mark.asyncio
    async def test_naming_a_prefill_member_is_refused_with_the_reason(self):
        # It would not fail: a prefill answers 200 after a single token. The
        # run would report latencies for a path no user request takes.
        model = _group_model()
        with _with_members(_group_members()):
            with pytest.raises(BadRequestException) as excinfo:
                await route._resolve_target_endpoint(None, model, "p")
        message = excinfo.value.message
        assert "router" in message
        assert "prefill" in message

    @pytest.mark.asyncio
    async def test_naming_the_router_is_accepted(self):
        model = _group_model()
        with _with_members(_group_members()):
            endpoint = await route._resolve_target_endpoint(None, model, "r")
        assert endpoint.name == "r"

    @pytest.mark.asyncio
    async def test_a_group_without_a_running_router_is_refused(self):
        # Not "fall back to a GPU member": no member of a group serves alone,
        # so there is nothing to measure and saying so is the honest answer.
        model = _group_model()
        members = [m for m in _group_members() if m.role != RoleNameEnum.ROUTER.value]
        with _with_members(members):
            with pytest.raises(BadRequestException) as excinfo:
                await route._resolve_target_endpoint(None, model, None)
        assert "router" in excinfo.value.message

    @pytest.mark.asyncio
    async def test_a_plain_model_resolves_to_its_running_instance(self):
        model = _plain_model()
        with _with_members([_member("mi", None, id=7)]):
            endpoint = await route._resolve_target_endpoint(None, model, None)
        assert endpoint.name == "mi"

    @pytest.mark.asyncio
    async def test_a_named_instance_of_a_plain_model_is_honored(self):
        # The behaviour every existing caller has: naming a replica picks it.
        model = _plain_model()
        members = [_member("a", None, id=1), _member("b", None, id=2)]
        with _with_members(members):
            endpoint = await route._resolve_target_endpoint(None, model, "b")
        assert endpoint.name == "b"

    @pytest.mark.asyncio
    async def test_an_instance_that_is_not_running_keeps_its_old_message(self):
        model = _plain_model()
        stopped = _member("mi", None, id=1, state=ModelInstanceStateEnum.ERROR)
        with _with_members([stopped]):
            with pytest.raises(BadRequestException) as excinfo:
                await route._resolve_target_endpoint(None, model, "mi")
        assert "not in RUNNING state" in excinfo.value.message


class TestPlacement:
    """`--processor` is a host path, so the run has to sit where the weights
    are — which under PD is never the endpoint."""

    @pytest.mark.asyncio
    async def test_a_group_runs_next_to_decode_not_the_router(self):
        model = _group_model()
        members = _group_members()
        router = members[-1]
        with _with_members(members):
            placement = await route._resolve_placement(None, model, router)
        assert placement.name == "d"
        assert placement.worker_id != router.worker_id

    @pytest.mark.asyncio
    async def test_prefill_is_used_when_there_is_no_decode(self):
        # Preference, not a requirement: the load generator is CPU-bound and
        # prefill is the role that spends its CPU on tokenization.
        model = _group_model()
        members = [m for m in _group_members() if m.role != RoleNameEnum.DECODE.value]
        with _with_members(members):
            placement = await route._resolve_placement(None, model, members[-1])
        assert placement.name == "p"

    @pytest.mark.asyncio
    async def test_no_member_holds_weights_falls_back_to_the_endpoint(self):
        model = _group_model()
        members = [
            _member("p", RoleNameEnum.PREFILL.value, id=1, worker_id=11, path=None),
            _member("r", RoleNameEnum.ROUTER.value, id=3, worker_id=13, path=None),
        ]
        with _with_members(members):
            placement = await route._resolve_placement(None, model, members[-1])
        assert placement.name == "r"

    @pytest.mark.asyncio
    async def test_a_plain_model_runs_where_it_always_did(self):
        model = _plain_model()
        instance = _member("mi", None, id=7)
        placement = await route._resolve_placement(None, model, instance)
        assert placement is instance


class TestSnapshot:
    """A group's endpoint is its router, which holds no accelerator: a snapshot
    of it alone reports a run on zero cards."""

    @pytest.mark.asyncio
    async def test_a_group_snapshot_covers_every_member(self):
        model = _group_model()
        members = _group_members()
        router = members[-1]
        with (
            _with_members(members),
            patch.object(
                route,
                "WorkerService",
                lambda _: SimpleNamespace(
                    get_by_id=AsyncMock(return_value=SimpleNamespace(name="w"))
                ),
            ),
            patch.object(route, "create_worker_snapshot", lambda *_: (None, None)),
        ):
            snapshot = await route.get_benchmark_snapshot(None, router, model)
        assert set(snapshot.instances) == {"p", "d", "r"}

    @pytest.mark.asyncio
    async def test_the_generation_measured_is_recorded(self):
        # Two runs of "the same model" measure different things across a
        # config edit; without this the reports cannot be told apart later.
        model = _group_model()
        members = _group_members()
        for m in members:
            # One group_id is one generation, so every member carries it.
            m.spec_digest = "abc123"
        with (
            _with_members(members),
            patch.object(
                route,
                "WorkerService",
                lambda _: SimpleNamespace(
                    get_by_id=AsyncMock(return_value=SimpleNamespace(name="w"))
                ),
            ),
            patch.object(route, "create_worker_snapshot", lambda *_: (None, None)),
        ):
            snapshot = await route.get_benchmark_snapshot(None, members[-1], model)
        assert snapshot.spec_digest == "abc123"

    @pytest.mark.asyncio
    async def test_a_plain_model_snapshot_is_unchanged(self):
        model = _plain_model()
        instance = _member("mi", None, id=7)
        with (
            _with_members([instance]),
            patch.object(
                route,
                "WorkerService",
                lambda _: SimpleNamespace(
                    get_by_id=AsyncMock(return_value=SimpleNamespace(name="w"))
                ),
            ),
            patch.object(route, "create_worker_snapshot", lambda *_: (None, None)),
        ):
            snapshot = await route.get_benchmark_snapshot(None, instance, model)
        assert set(snapshot.instances) == {"mi"}


class TestTargetModel:
    @pytest.mark.asyncio
    async def test_an_instance_name_still_identifies_its_model(self):
        # The instance list page's "run benchmark" action names a member.
        instance = _member("d", RoleNameEnum.DECODE.value, id=2)
        model = _group_model()
        with (
            patch.object(
                ModelInstance, "one_by_field", AsyncMock(return_value=instance)
            ),
            patch.object(Model, "one_by_id", AsyncMock(return_value=model)),
        ):
            resolved = await route._resolve_target_model(
                None, BenchmarkCreate(name="bm", model_instance_name="d")
            )
        assert resolved.name == "g"

    @pytest.mark.asyncio
    async def test_naming_nothing_at_all_is_refused(self):
        with pytest.raises(BadRequestException) as excinfo:
            await route._resolve_target_model(None, BenchmarkCreate(name="bm"))
        assert "model_id" in excinfo.value.message


class TestRouteMode:
    """`instance` measures an engine, `route` measures a deployment.

    The distinction is not overhead: a plain model with four replicas is four
    replicas through its route and exactly one straight at a member, which is
    why comparing a 2P2D group against a four-replica deployment needs this
    mode on both sides.
    """

    def _targets(self, *, active=True, route_id=5):
        return [
            SimpleNamespace(
                id=1,
                model_id=1,
                route_id=route_id,
                route_name="qwen",
                state=(
                    TargetStateEnum.ACTIVE if active else TargetStateEnum.UNAVAILABLE
                ),
            )
        ]

    async def _resolve(self, model, targets, owner):
        with (
            patch.object(
                route.ModelRouteTarget,
                "all_by_fields",
                AsyncMock(return_value=targets),
            ),
            patch.object(
                route.ModelRoute,
                "one_by_id",
                AsyncMock(
                    return_value=SimpleNamespace(name="qwen", owner_principal_id=9)
                ),
            ),
            patch.object(route.Principal, "one_by_id", AsyncMock(return_value=owner)),
        ):
            return await route._resolve_route_name(None, model)

    @pytest.mark.asyncio
    async def test_the_platform_orgs_route_keeps_its_bare_name(self):
        # What existing clients call it, and what the proxy matches on.
        owner = SimpleNamespace(id=route.platform_principal_id(), name="platform")
        assert await self._resolve(_plain_model(), self._targets(), owner) == "qwen"

    @pytest.mark.asyncio
    async def test_another_orgs_route_is_prefixed(self):
        owner = SimpleNamespace(id=42, name="org1")
        assert await self._resolve(_plain_model(), self._targets(), owner) == (
            "org1/qwen"
        )

    @pytest.mark.asyncio
    async def test_no_active_route_is_refused_rather_than_measured(self):
        # An UNAVAILABLE target resolves to nothing at request time, so the run
        # would measure a 503 and report it as the deployment's latency.
        owner = SimpleNamespace(id=42, name="org1")
        with pytest.raises(BadRequestException) as excinfo:
            await self._resolve(_plain_model(), [], owner)
        assert "route" in excinfo.value.message

    def test_instance_is_the_default(self):
        # A create body that names no target mode measures an engine.
        assert (
            BenchmarkCreate(name="bm", model_id=1).target_mode
            is BenchmarkTargetModeEnum.INSTANCE
        )


class TestANamedRouteIsChecked:
    """A model can sit behind several routes, and they are not the same
    measurement: a canary sends a share of the load to a different model
    entirely. So the form names the route it listed, and the pairing is
    verified rather than trusted."""

    def _targets(self, route_ids):
        return [
            SimpleNamespace(
                id=i + 1,
                model_id=1,
                route_id=rid,
                route_name=f"r{rid}",
                state=TargetStateEnum.ACTIVE,
            )
            for i, rid in enumerate(route_ids)
        ]

    async def _resolve(self, requested, route_ids=(5, 6)):
        routes = {
            5: SimpleNamespace(name="alias", owner_principal_id=9),
            6: SimpleNamespace(name="canary", owner_principal_id=9),
        }
        with (
            patch.object(
                route.ModelRouteTarget,
                "all_by_fields",
                AsyncMock(return_value=self._targets(route_ids)),
            ),
            patch.object(
                route.ModelRoute,
                "one_by_id",
                AsyncMock(side_effect=lambda _s, rid: routes.get(rid)),
            ),
            patch.object(
                route.Principal,
                "one_by_id",
                AsyncMock(
                    return_value=SimpleNamespace(
                        id=route.platform_principal_id(), name="platform"
                    )
                ),
            ),
        ):
            return await route._resolve_named_route(None, _plain_model(), requested)

    @pytest.mark.asyncio
    async def test_the_named_route_is_the_one_used(self):
        # Not "the first active one": picking for the user would measure
        # whichever route the server happened to choose.
        assert await self._resolve("canary") == "canary"

    @pytest.mark.asyncio
    async def test_a_route_that_does_not_front_this_model_is_refused(self):
        with pytest.raises(BadRequestException) as excinfo:
            await self._resolve("someone-elses-route")
        assert "does not have an active target" in excinfo.value.message

    @pytest.mark.asyncio
    async def test_the_prefixed_and_bare_names_both_resolve(self):
        # The form may hold either, and both address the same route.
        assert await self._resolve("alias", route_ids=(5,)) == "alias"
