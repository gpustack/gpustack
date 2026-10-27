"""Offline tests for the Shuihua provider.

Nothing here touches the network: the pure helpers take plain inputs, and the
few cases that must exercise a request path inject an ``httpx.MockTransport``,
which answers in-process. Response bodies are the shapes the live API was
observed to return.
"""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import httpx
import pytest

from gpustack.cloud_providers.abstract import (
    CloudInstanceCreate,
    InstanceProvisioningFailed,
    InstanceState,
)
from gpustack.cloud_providers.common import creation_idempotency_key
from gpustack.cloud_providers.shuihua import (
    MAX_IDEMPOTENCY_KEY_SIZE,
    MAX_INSTANCE_NAME_SIZE,
    MAX_USER_DATA_SIZE,
    PENDING_STATUSES,
    TERMINAL_STATES,
    ShuihuaAPIError,
    ShuihuaClient,
    ShuihuaProviderClient,
    VMDetail,
    VMStatus,
    _idempotency_key,
    _parse_vm_id,
    _raise_if_terminal,
    _to_cloud_instance,
    status_mapping,
)

# The body GET /vms/{id} returned for an active instance created with
# user_data, mapping only SSH and HTTP.
ACTIVE_VM = {
    "id": 33,
    "template_id": 1,
    "template_name": "RTX4090-16C60G",
    "instance_id": "49a1d649-9053-4fca-a746-1cb7ce348701",
    "instance_name": "pool-3-abc",
    "ip_address": "172.16.46.22",
    "port_mappings": [
        {"public_ip": "125.67.215.17", "public_port": 33004, "internal_port": 22},
        {"public_ip": "125.67.215.17", "public_port": 33005, "internal_port": 80},
    ],
    "status": "active",
    "ssh_user": "ubuntu",
    "ssh_private_key": "",
}


def transport(handler):
    """A client whose requests are answered in-process."""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def responder(status, body, capture=None):
    def handle(request):
        if capture is not None:
            capture["body"] = json.loads(request.content) if request.content else None
            capture["url"] = str(request.url)
        return httpx.Response(status, json=body)

    return handle


def provider(handler):
    """A provider client wired to an in-process transport."""
    client = ShuihuaClient(
        api_key="k", base_url="https://api.test", client=transport(handler)
    )
    p = ShuihuaProviderClient(api_key="k", base_url="https://api.test")
    p._api = lambda: _as_ctx(client)
    return p


class _as_ctx:
    def __init__(self, client):
        self._client = client

    async def __aenter__(self):
        return self._client

    async def __aexit__(self, *_):
        return False


# --------------------------------------------------------------------------
# instance mapping
# --------------------------------------------------------------------------


def test_to_cloud_instance_advertises_the_private_address():
    """One public IP fronts every instance, so it cannot identify one.

    Only ports 22 and 80 are mapped and never the worker's own, so nothing
    reaches these workers inbound anyway; the per-instance private address is
    what distinguishes them.
    """
    instance = _to_cloud_instance(VMDetail(**ACTIVE_VM))

    assert instance.ip_address == "172.16.46.22"
    assert instance.ssh_endpoint == ("125.67.215.17", 33004)
    assert instance.ssh_user == "ubuntu"
    assert instance.status == InstanceState.RUNNING
    assert instance.external_id == "33"


def test_to_cloud_instance_without_a_mapping():
    """An instance the API reports no mapping for yields no SSH endpoint."""
    vm = VMDetail(
        **{
            "id": 29,
            "ip_address": "172.16.46.11",
            "internal_port": 22,
            "status": "active",
        }
    )
    instance = _to_cloud_instance(vm)

    assert instance.ip_address == "172.16.46.11"
    assert instance.ssh_endpoint is None
    assert instance.ssh_user is None


def test_public_endpoint_resolves_only_mapped_ports():
    vm = VMDetail(**ACTIVE_VM)

    assert vm.public_endpoint(22) == ("125.67.215.17", 33004)
    assert vm.public_endpoint(80) == ("125.67.215.17", 33005)
    # The worker's own port is never mapped, which is why these clusters can
    # only be reached through the tunnel.
    assert vm.public_endpoint(10150) is None


@pytest.mark.parametrize(
    "status, expected",
    [
        ("creating", InstanceState.CREATED),
        ("processing", InstanceState.CREATED),
        ("active", InstanceState.RUNNING),
        ("failed", InstanceState.FAILED),
        ("expired", InstanceState.STOPPED),
        ("terminated", InstanceState.TERMINATED),
    ],
)
def test_status_mapping_is_total(status, expected):
    assert status_mapping[VMStatus(status)] is expected


def test_pending_and_terminal_sets_do_not_overlap():
    pending = {status_mapping[s] for s in PENDING_STATUSES}
    assert not pending & set(TERMINAL_STATES)


def test_unknown_status_degrades_instead_of_failing_the_parse():
    """The status set grew by three values between API 1.3.0 and 1.4.0.

    A strict enum would turn the next addition into a hard error for the whole
    list_vms page the value appears on, taking every worker on the credential
    with it.
    """
    vm = VMDetail(**{"id": 1, "status": "some-new-status"})

    assert vm.status is None
    assert _to_cloud_instance(vm).status is InstanceState.UNKNOWN


# --------------------------------------------------------------------------
# terminal-lease guards
# --------------------------------------------------------------------------


@pytest.mark.parametrize("status", ["failed", "terminated"])
def test_raise_if_terminal_gives_up_on_a_dead_lease(status):
    """Replaying a spent key returns the dead lease, so waiting is futile."""
    instance = _to_cloud_instance(VMDetail(**{"id": 7, "status": status}))

    with pytest.raises(InstanceProvisioningFailed):
        _raise_if_terminal("7", instance)


@pytest.mark.parametrize("status", ["creating", "processing", "active", "expired"])
def test_raise_if_terminal_keeps_waiting_otherwise(status):
    # expired is only powered off and the API offers a start endpoint, so it is
    # not provably dead.
    instance = _to_cloud_instance(VMDetail(**{"id": 7, "status": status}))

    _raise_if_terminal("7", instance)


def test_raise_if_terminal_tolerates_a_missing_instance():
    _raise_if_terminal("7", None)


@pytest.mark.parametrize("value", ["", "abc", "12x", None])
def test_parse_vm_id_rejects_what_cannot_name_a_vm(value):
    """Every id handed out is a stringified int, so anything else matches none.

    Returning None keeps a ValueError from escaping into the provisioning
    controller, where it would strand the worker on an id that can never
    resolve.
    """
    assert _parse_vm_id(value) is None


def test_parse_vm_id_accepts_an_integer_string():
    assert _parse_vm_id("33") == 33


# --------------------------------------------------------------------------
# idempotency key
# --------------------------------------------------------------------------


def _worker(updated_at, worker_id=42, name="pool-3-abc", cluster_id=3):
    worker = MagicMock()
    worker.id = worker_id
    worker.name = name
    worker.cluster_id = cluster_id
    worker.updated_at = updated_at
    return worker


PG_PRECISION = datetime(2026, 8, 21, 3, 7, 51, 138582, tzinfo=timezone.utc)
MYSQL_PRECISION = datetime(2026, 8, 21, 3, 7, 51, tzinfo=timezone.utc)
NAIVE_UTC = datetime(2026, 8, 21, 3, 7, 51)
LATER = datetime(2026, 8, 21, 3, 9, 0, tzinfo=timezone.utc)


def test_idempotency_key_is_stable_across_reads():
    """Two reads of the same row must agree, or a retry provisions a second VM."""
    assert creation_idempotency_key(_worker(PG_PRECISION)) == creation_idempotency_key(
        _worker(PG_PRECISION)
    )


def test_idempotency_key_ignores_timestamp_precision():
    """MySQL stores TIMESTAMP without fractional digits, PostgreSQL keeps them."""
    assert creation_idempotency_key(_worker(PG_PRECISION)) == creation_idempotency_key(
        _worker(MYSQL_PRECISION)
    )


def test_idempotency_key_reads_a_naive_timestamp_as_utc():
    """The column holds naive UTC; reading it as local time would shift it."""
    assert creation_idempotency_key(_worker(NAIVE_UTC)) == creation_idempotency_key(
        _worker(PG_PRECISION)
    )


def test_idempotency_key_turns_over_when_the_row_is_rewritten():
    """A write that re-drives the worker has to mint a new key."""
    assert creation_idempotency_key(_worker(PG_PRECISION)) != creation_idempotency_key(
        _worker(LATER)
    )


def test_idempotency_key_is_per_worker():
    assert creation_idempotency_key(_worker(PG_PRECISION)) != creation_idempotency_key(
        _worker(PG_PRECISION, worker_id=43)
    )


def test_idempotency_key_fits_the_api_limit():
    key = creation_idempotency_key(_worker(PG_PRECISION))

    assert 0 < len(key) <= MAX_IDEMPOTENCY_KEY_SIZE


def test_idempotency_key_survives_a_worker_without_a_timestamp():
    assert creation_idempotency_key(_worker(None))


def _create(**kwargs):
    defaults = {
        "name": "pool-3-abc",
        "image": "img-uuid",
        "type": "1",
        "region": "",
        "user_data": "#cloud-config\nruncmd: []\n",
        "labels": {"cluster_id": 3, "worker_id": 42},
    }
    return CloudInstanceCreate(**{**defaults, **kwargs})


def test_provider_prefers_the_caller_supplied_key():
    assert _idempotency_key(_create(idempotency_key="from-the-worker-row")) == (
        "from-the-worker-row"
    )


def test_provider_derives_a_key_when_none_was_supplied():
    """Only a hand-built CloudInstanceCreate hits this; the API demands a key."""
    key = _idempotency_key(_create())

    assert key and len(key) <= MAX_IDEMPOTENCY_KEY_SIZE
    assert key == _idempotency_key(_create())


# --------------------------------------------------------------------------
# request path (in-process transport)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_unwraps_the_data_envelope():
    client = ShuihuaClient(
        api_key="k",
        base_url="https://api.test",
        client=transport(
            responder(200, {"data": [{"template_id": 1, "gpu_model": "RTX4090"}]})
        ),
    )

    specs = await client.list_gpu_instances()

    assert [s.template_id for s in specs] == [1]
    assert specs[0].gpu_model == "RTX4090"


@pytest.mark.asyncio
async def test_request_raises_on_the_error_envelope():
    client = ShuihuaClient(
        api_key="k",
        base_url="https://api.test",
        client=transport(
            responder(404, {"error": {"code": "NOT_FOUND", "message": "虚拟机不存在"}})
        ),
    )

    with pytest.raises(ShuihuaAPIError) as exc:
        await client.get_vm(999)

    assert exc.value.status_code == 404
    assert exc.value.code == "NOT_FOUND"
    assert exc.value.message == "虚拟机不存在"


@pytest.mark.asyncio
async def test_get_instance_maps_a_missing_vm_to_none():
    p = provider(responder(404, {"error": {"code": "NOT_FOUND", "message": "gone"}}))

    assert await p.get_instance("33") is None


@pytest.mark.asyncio
async def test_get_instance_skips_the_call_for_an_unusable_id():
    def explode(request):  # pragma: no cover - must not be reached
        raise AssertionError("no request should be sent")

    assert await provider(explode).get_instance("not-an-id") is None


@pytest.mark.asyncio
async def test_iter_vms_walks_past_the_first_page_without_total_pages():
    """total_pages is optional, so it cannot be the only stop condition.

    Coercing a missing one to 0 ended the walk after page 1, hiding every VM
    but the first page's.
    """
    pages = {
        1: {"data": [{"id": 1}, {"id": 2}], "meta": {}},
        2: {"data": [{"id": 3}], "meta": {}},
    }

    def handle(request):
        page = int(dict(request.url.params)["page"])
        return httpx.Response(200, json=pages[page])

    client = ShuihuaClient(
        api_key="k", base_url="https://api.test", client=transport(handle)
    )

    assert [vm.id async for vm in client.iter_vms(page_size=2)] == [1, 2, 3]


@pytest.mark.asyncio
async def test_iter_vms_stops_on_the_page_count_when_given_one():
    def handle(request):
        page = int(dict(request.url.params)["page"])
        assert page == 1, "should not ask for a page beyond total_pages"
        return httpx.Response(
            200, json={"data": [{"id": 1}], "meta": {"total_pages": 1}}
        )

    client = ShuihuaClient(
        api_key="k", base_url="https://api.test", client=transport(handle)
    )

    assert [vm.id async for vm in client.iter_vms(page_size=1)] == [1]


@pytest.mark.asyncio
async def test_create_vm_sends_the_replay_guard_and_trims_the_image_id():
    """The image list has been seen handing back uuids with surrounding space."""
    captured = {}
    client = ShuihuaClient(
        api_key="k",
        base_url="https://api.test",
        client=transport(
            responder(202, {"data": {"id": 29, "status": "creating"}}, captured)
        ),
    )

    vm = await client.create_vm(
        template_id=1,
        image_id="  img-uuid  ",
        idempotency_key="token",
        user_data="#cloud-config\n",
        instance_name="pool-3-abc",
    )

    assert captured["body"] == {
        "template_id": 1,
        "image_id": "img-uuid",
        "idempotency_key": "token",
        "user_data": "#cloud-config\n",
        "instance_name": "pool-3-abc",
    }
    # Creation is asynchronous: accepted now, polled to completion later.
    assert vm.status is VMStatus.CREATING
    assert vm.status in PENDING_STATUSES


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        {"idempotency_key": ""},
        {"idempotency_key": "x" * (MAX_IDEMPOTENCY_KEY_SIZE + 1)},
        {"user_data": "not-cloud-config"},
        {"user_data": "#cloud-config\n" + "x" * MAX_USER_DATA_SIZE},
        {"instance_name": "n" * (MAX_INSTANCE_NAME_SIZE + 1)},
    ],
)
async def test_create_vm_validates_before_spending_money(kwargs):
    """Creation is billed on acceptance, so bad input must not reach the API."""

    def explode(request):  # pragma: no cover - must not be reached
        raise AssertionError("no request should be sent")

    client = ShuihuaClient(
        api_key="k", base_url="https://api.test", client=transport(explode)
    )
    call = {
        "template_id": 1,
        "image_id": "img",
        "idempotency_key": "token",
        "user_data": "#cloud-config\n",
        **kwargs,
    }

    with pytest.raises(ValueError):
        await client.create_vm(**call)


@pytest.mark.asyncio
async def test_create_instance_returns_the_accepted_lease_id():
    p = provider(responder(202, {"data": {"id": 29, "status": "creating"}}))

    assert await p.create_instance(_create(idempotency_key="token")) == "29"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["terminated", "failed"])
async def test_create_instance_rejects_a_replayed_dead_lease(status):
    """Replaying a spent key hands back that lease rather than making a VM.

    Only a new key can start over, so there is nothing here to wait for.
    """
    p = provider(responder(202, {"data": {"id": 29, "status": status}}))

    with pytest.raises(InstanceProvisioningFailed):
        await p.create_instance(_create(idempotency_key="spent"))


@pytest.mark.asyncio
async def test_create_instance_requires_user_data():
    def explode(request):  # pragma: no cover - must not be reached
        raise AssertionError("no request should be sent")

    with pytest.raises(ValueError):
        await provider(explode).create_instance(_create(user_data=None))


@pytest.mark.asyncio
async def test_create_instance_requires_a_numeric_template():
    def explode(request):  # pragma: no cover - must not be reached
        raise AssertionError("no request should be sent")

    with pytest.raises(ValueError):
        await provider(explode).create_instance(_create(type="not-a-template"))


@pytest.mark.asyncio
async def test_delete_instance_tolerates_a_vanished_vm():
    p = provider(responder(404, {"error": {"code": "NOT_FOUND", "message": "gone"}}))

    await p.delete_instance("33")


@pytest.mark.asyncio
async def test_delete_instance_surfaces_other_failures():
    """400 is any bad request, plausibly "cannot terminate while creating".

    Swallowing it would delete the worker row while a live, billing VM stays up
    with nothing left pointing at it.
    """
    p = provider(responder(400, {"error": {"code": "BAD_STATE", "message": "busy"}}))

    with pytest.raises(ShuihuaAPIError):
        await p.delete_instance("33")


@pytest.mark.asyncio
async def test_volumes_are_rejected_rather_than_dropped():
    """There is no block storage API, so a configured volume cannot be honoured."""
    p = provider(responder(200, {"data": {}}))

    with pytest.raises(NotImplementedError):
        await p.create_volumes_and_attach(1, "33", "", MagicMock())

    assert await p.create_volumes_and_attach(1, "33", "") == []
