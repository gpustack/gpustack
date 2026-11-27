from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import aiohttp
import pytest

from gpustack.client import worker_filesystem_client as module


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [404, 403, 500])
async def test_optional_model_config_distinguishes_absence_from_failure(
    monkeypatch, status
):
    response = Mock(status=status)
    response.raise_for_status.side_effect = aiohttp.ClientResponseError(
        request_info=SimpleNamespace(real_url="http://worker/files/model-config"),
        history=(),
        status=status,
    )
    request = AsyncMock(return_value=(response, b'{}'))
    monkeypatch.setattr(module, "request_to_worker", request)
    client = module.WorkerFilesystemClient.__new__(module.WorkerFilesystemClient)
    client._http_client = object()
    client._http_client_no_proxy = object()
    if status == 404:
        assert (
            await client.read_model_config(
                object(), "/model/model_index.json", missing_ok=True
            )
            is None
        )
        response.raise_for_status.assert_not_called()
    else:
        with pytest.raises(aiohttp.ClientResponseError) as error:
            await client.read_model_config(
                object(), "/model/model_index.json", missing_ok=True
            )
        assert error.value.status == status
    assert request.call_args.kwargs["raise_on_error"] is False
