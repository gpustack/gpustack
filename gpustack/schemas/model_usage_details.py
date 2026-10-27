from datetime import date, datetime
from typing import ClassVar, Optional

from pydantic import ConfigDict
from sqlalchemy import BigInteger, Boolean, Column, Integer, false
from sqlmodel import Field, SQLModel

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import UTCDateTime
from gpustack.schemas.model_usage import OperationEnum


class ModelUsageDetails(SQLModel, BaseModelMixin, table=True):
    """
    Per-request inference usage audit row.

    Reference id columns (``user_id`` / ``model_id`` / ``model_route_id`` /
    ``provider_id`` / ``cluster_id`` / ``api_key_id``) are plain integers,
    not foreign keys. Audit rows must outlive the entities they describe;
    losing the historical id (which ``SET NULL`` would do on parent delete)
    is a worse audit outcome than losing the live join, so ids stay as
    reported and ``*_name`` columns hold mutable display snapshots
    alongside.

    Relationship to ``ModelUsage``: details and rollup are NOT 1:1 — the
    rollup aggregates many requests per (model, user, key, operation, day)
    into one row, while details preserves every report. They are populated
    from the same ingest path but serve different read patterns:
        * ``ModelUsage``    — dashboard / per-day analytics, FK-friendly
        * ``ModelUsageDetails`` — quota reconciliation / per-request audit,
                                  FK-less so historical ids survive deletes

    Both rows are constructed with the same ``build_model_usage_snapshot``
    keys plus table-specific extras; see that helper's docstring for the
    shared-snapshot contract.
    """

    __tablename__: ClassVar[str] = "model_usage_details"
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    user_name: Optional[str] = Field(default=None)
    model_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    model_name: str = Field(default=...)
    model_route_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    model_route_name: Optional[str] = Field(default=None)
    # Tenant scope snapshot. FK-less for the same audit-survival reason
    # as the other id columns — see class docstring.
    owner_principal_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    # Consumer tenant scope snapshot, denormalized from the API key owner.
    consumer_principal_id: Optional[int] = Field(
        default=None, sa_column=Column(Integer)
    )
    provider_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    provider_name: Optional[str] = Field(default=None)
    provider_type: Optional[str] = Field(default=None)
    cluster_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    cluster_name: Optional[str] = Field(default=None)
    api_key_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    api_key_name: Optional[str] = Field(default=None)
    access_key: Optional[str] = Field(default=None)
    api_key_is_custom: Optional[bool] = Field(default=None)
    date: date
    prompt_token_count: int = Field(
        default=..., sa_column=Column(BigInteger, nullable=False)
    )
    completion_token_count: int = Field(
        default=..., sa_column=Column(BigInteger, nullable=False)
    )
    prompt_cached_token_count: int = Field(
        default=0, sa_column=Column(BigInteger, nullable=False, default=0)
    )
    # True iff the canonical usage chunk was observed before the stream
    # ended (token counts above are authoritative). False means the request
    # was interrupted and the token counts are server-side estimates (see
    # ``_estimate_partial_usage``). Billing reads this to gate per-request
    # charges — image / tts / stt are billed per request and have no token
    # fallback, so an interrupted one must not be charged — and to flag
    # estimated token-billed rows for reconciliation / transparency.
    completed: bool = Field(
        default=False,
        sa_column=Column(Boolean, nullable=False, server_default=false()),
    )
    operation: Optional[OperationEnum] = Field(default=None)
    # Wall-clock anchors reported by the proxy (UnixMilli on the wire,
    # stored as naive UTC). Distinct from ``created_at`` so quota
    # reconciliation / cache rebuild can key off the request's actual
    # completion time even after rows are archived or migrated.
    started_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime(), nullable=True)
    )
    completed_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime(), nullable=True)
    )
    # Milliseconds from request entry to the first response body *chunk*, as
    # reported by the proxy. Streaming only — the non-streaming path never
    # sees intermediate chunks. Deliberately the same figure the client is
    # handed as ``usage.time_to_first_token_ms`` rather than a stricter
    # time-to-first-token: for OpenAI the first chunk is usually a role-only
    # delta, so this runs a few milliseconds optimistic, and two numbers under
    # one name that disagree would be worse.
    #
    # There is no ``duration_ms`` beside it on purpose: the duration is
    # ``completed_at - started_at``, both of which are already here, and a
    # third number could only be one that disagrees with them.
    ttft_ms: Optional[int] = Field(default=None, sa_column=Column(Integer))
    # Envoy's ``x-request-id``, which is what makes it the right column to key
    # an audit lookup on: it exists for every tracked request whatever the
    # endpoint or outcome, and it is the value the Envoy access log already
    # carries, so one id quoted by a user resolves in both places. The gateway
    # echoes it downstream under ``X-GPUStack-Request-Id`` so the user has it
    # to quote.
    #
    # Indexed, never constrained unique. It identifies a *downstream* request
    # while a row is written per filter-chain run, and a fallback pass is an
    # internal redirect of the same downstream request — so two rows can
    # legitimately carry one value.
    request_id: Optional[str] = Field(default=None, index=True)
    # The model's own id for this response (``chatcmpl-…``, ``resp_…``,
    # ``msg_…``, ``embd-…``), taken verbatim from whatever the upstream put
    # there. This is the id a caller reads off an SDK response, and the only
    # one a third-party provider can be asked about. NULL when the upstream
    # mints none, when the endpoint returns no JSON at all (TTS, image), and
    # when no response arrived.
    upstream_response_id: Optional[str] = Field(default=None)

    model_config = ConfigDict(protected_namespaces=())


class ModelUsageDetailsArchive(SQLModel, BaseModelMixin, table=True):
    """
    Cold-storage archive for ``model_usage_details``.

    Same column layout as the hot table; ``id`` is a plain primary key with
    no sequence/autoincrement — rows are archived from
    ``model_usage_details`` and reuse the source ``id``.
    """

    __tablename__: ClassVar[str] = "model_usage_details_archive"
    id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, primary_key=True, autoincrement=False),
    )
    user_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    user_name: Optional[str] = Field(default=None)
    model_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    model_name: str = Field(default=...)
    model_route_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    model_route_name: Optional[str] = Field(default=None)
    # Tenant scope snapshot. FK-less for the same audit-survival reason
    # as the other id columns — see class docstring.
    owner_principal_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    # Consumer tenant scope snapshot, denormalized from the API key owner.
    consumer_principal_id: Optional[int] = Field(
        default=None, sa_column=Column(Integer)
    )
    provider_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    provider_name: Optional[str] = Field(default=None)
    provider_type: Optional[str] = Field(default=None)
    cluster_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    cluster_name: Optional[str] = Field(default=None)
    api_key_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    api_key_name: Optional[str] = Field(default=None)
    access_key: Optional[str] = Field(default=None)
    api_key_is_custom: Optional[bool] = Field(default=None)
    date: date
    prompt_token_count: int = Field(
        default=..., sa_column=Column(BigInteger, nullable=False)
    )
    completion_token_count: int = Field(
        default=..., sa_column=Column(BigInteger, nullable=False)
    )
    prompt_cached_token_count: int = Field(
        default=0, sa_column=Column(BigInteger, nullable=False, default=0)
    )
    # True iff the canonical usage chunk was observed before the stream
    # ended (token counts above are authoritative). False means the request
    # was interrupted and the token counts are server-side estimates (see
    # ``_estimate_partial_usage``). Billing reads this to gate per-request
    # charges — image / tts / stt are billed per request and have no token
    # fallback, so an interrupted one must not be charged — and to flag
    # estimated token-billed rows for reconciliation / transparency.
    completed: bool = Field(
        default=False,
        sa_column=Column(Boolean, nullable=False, server_default=false()),
    )
    operation: Optional[OperationEnum] = Field(default=None)
    started_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime(), nullable=True)
    )
    completed_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime(), nullable=True)
    )
    ttft_ms: Optional[int] = Field(default=None, sa_column=Column(Integer))
    # Indexed here as well as on the hot table. An id a user quotes is most
    # often one from a request old enough to have been archived, so a lookup
    # that only worked before archival would be the wrong half of the feature.
    request_id: Optional[str] = Field(default=None, index=True)
    upstream_response_id: Optional[str] = Field(default=None)

    model_config = ConfigDict(protected_namespaces=())
