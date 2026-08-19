"""Unit tests for :func:`gpustack.routes.gpu_instances_helper.handle_error`.

The helper is the single translation point between the ``worker.gpustack.ai/v1``
CRD client's ``ApiException`` and the project's HTTP exceptions. It is shared by
``gpu_instance_types`` (6 call sites) and ``gpu_instance_type_flavors`` (1), so a
mis-mapping here reaches every write those routes proxy into a cluster.

The 502/503/504 branch is the fix for
https://github.com/gpustack/gpustack/issues/6071: a write against a cluster with
no ready worker gets a 503 from the cluster proxy, and folding it into the 500
fallback produced a response that contradicted itself —
``{"code": 500, "message": "Service Unavailable", "reason": "InternalServerError"}``.
The remaining cases characterize the mappings that branch must not disturb.
"""

import http

import pytest
from kubernetes_asyncio import client

from gpustack.api.exceptions import (
    AlreadyExistsException,
    InternalServerErrorException,
    InvalidException,
    NotFoundException,
    ServiceUnavailableException,
)
from gpustack.routes.gpu_instances_helper import handle_error


def _api_exception(status: int, reason: str = "boom") -> client.exceptions.ApiException:
    return client.exceptions.ApiException(status=status, reason=reason)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [
        http.HTTPStatus.BAD_GATEWAY,
        http.HTTPStatus.SERVICE_UNAVAILABLE,
        http.HTTPStatus.GATEWAY_TIMEOUT,
    ],
)
async def test_upstream_unavailable_maps_to_service_unavailable(status):
    """#6071: an unreachable cluster reports its real cause, not a blanket 500."""
    with pytest.raises(ServiceUnavailableException) as excinfo:
        async with handle_error():
            raise _api_exception(status, reason="Service Unavailable")

    assert excinfo.value.status_code == 503
    assert excinfo.value.reason == "ServiceUnavailable"
    # The upstream reason is carried through, the same way every other branch
    # does it — the message is what told #6071's reporter the real cause.
    assert excinfo.value.message == "Service Unavailable"


@pytest.mark.asyncio
async def test_not_found_maps_to_not_found():
    with pytest.raises(NotFoundException) as excinfo:
        async with handle_error():
            raise _api_exception(http.HTTPStatus.NOT_FOUND, reason="Not Found")

    assert excinfo.value.status_code == 404
    assert excinfo.value.message == "Not Found"


@pytest.mark.asyncio
async def test_conflict_maps_to_already_exists():
    with pytest.raises(AlreadyExistsException) as excinfo:
        async with handle_error():
            raise _api_exception(http.HTTPStatus.CONFLICT, reason="Conflict")

    assert excinfo.value.status_code == 409
    assert excinfo.value.message == "Conflict"


@pytest.mark.asyncio
async def test_bad_request_maps_to_invalid():
    # Asserted on the exception type, not the status code: ``InvalidException``
    # is 422, which is a pre-existing choice this change does not revisit.
    with pytest.raises(InvalidException) as excinfo:
        async with handle_error():
            raise _api_exception(http.HTTPStatus.BAD_REQUEST, reason="Bad Request")

    assert excinfo.value.message == "Bad Request"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [
        http.HTTPStatus.INTERNAL_SERVER_ERROR,
        http.HTTPStatus.IM_A_TEAPOT,
    ],
)
async def test_unmapped_status_falls_back_to_internal_server_error(status):
    """Everything outside the mapped statuses keeps the 500 fallback."""
    with pytest.raises(InternalServerErrorException) as excinfo:
        async with handle_error():
            raise _api_exception(status)

    assert excinfo.value.status_code == 500
    assert excinfo.value.message == "boom"


@pytest.mark.asyncio
async def test_a_non_api_exception_propagates_untouched():
    """The helper only translates ``ApiException``; anything else is not its
    business and must reach the caller unchanged."""
    boom = RuntimeError("not a kubernetes failure")

    with pytest.raises(RuntimeError) as excinfo:
        async with handle_error():
            raise boom

    assert excinfo.value is boom
