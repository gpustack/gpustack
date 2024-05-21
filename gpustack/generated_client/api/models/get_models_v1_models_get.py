from http import HTTPStatus
from typing import Any, Dict, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.paginated_list_model_public import PaginatedListModelPublic
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    query: Union[None, Unset, str] = UNSET,
    page: Union[Unset, int] = 1,
    per_page: Union[Unset, int] = 100,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {}

    json_query: Union[None, Unset, str]
    if isinstance(query, Unset):
        json_query = UNSET
    else:
        json_query = query
    params["query"] = json_query

    params["page"] = page

    params["perPage"] = per_page

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: Dict[str, Any] = {
        "method": "get",
        "url": "/v1/models",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, PaginatedListModelPublic]]:
    if response.status_code == HTTPStatus.OK:
        response_200 = PaginatedListModelPublic.from_dict(response.json())

        return response_200
    if response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[Union[HTTPValidationError, PaginatedListModelPublic]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    query: Union[None, Unset, str] = UNSET,
    page: Union[Unset, int] = 1,
    per_page: Union[Unset, int] = 100,
) -> Response[Union[HTTPValidationError, PaginatedListModelPublic]]:
    """Get Models

    Args:
        query (Union[None, Unset, str]):
        page (Union[Unset, int]):  Default: 1.
        per_page (Union[Unset, int]):  Default: 100.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, PaginatedListModelPublic]]
    """

    kwargs = _get_kwargs(
        query=query,
        page=page,
        per_page=per_page,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    query: Union[None, Unset, str] = UNSET,
    page: Union[Unset, int] = 1,
    per_page: Union[Unset, int] = 100,
) -> Optional[Union[HTTPValidationError, PaginatedListModelPublic]]:
    """Get Models

    Args:
        query (Union[None, Unset, str]):
        page (Union[Unset, int]):  Default: 1.
        per_page (Union[Unset, int]):  Default: 100.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, PaginatedListModelPublic]
    """

    return sync_detailed(
        client=client,
        query=query,
        page=page,
        per_page=per_page,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    query: Union[None, Unset, str] = UNSET,
    page: Union[Unset, int] = 1,
    per_page: Union[Unset, int] = 100,
) -> Response[Union[HTTPValidationError, PaginatedListModelPublic]]:
    """Get Models

    Args:
        query (Union[None, Unset, str]):
        page (Union[Unset, int]):  Default: 1.
        per_page (Union[Unset, int]):  Default: 100.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, PaginatedListModelPublic]]
    """

    kwargs = _get_kwargs(
        query=query,
        page=page,
        per_page=per_page,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    query: Union[None, Unset, str] = UNSET,
    page: Union[Unset, int] = 1,
    per_page: Union[Unset, int] = 100,
) -> Optional[Union[HTTPValidationError, PaginatedListModelPublic]]:
    """Get Models

    Args:
        query (Union[None, Unset, str]):
        page (Union[Unset, int]):  Default: 1.
        per_page (Union[Unset, int]):  Default: 100.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, PaginatedListModelPublic]
    """

    return (
        await asyncio_detailed(
            client=client,
            query=query,
            page=page,
            per_page=per_page,
        )
    ).parsed
