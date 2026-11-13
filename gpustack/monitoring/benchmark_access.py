"""Authorization shared by live Benchmark monitoring and archive replay."""

from gpustack.api.exceptions import ForbiddenException
from gpustack.api.tenant import assert_resource_visible
from gpustack.schemas.principals import OrgRole, PrincipalType


def require_benchmark_monitoring_user(ctx):
    if ctx is None or getattr(ctx.user, 'kind', None) == PrincipalType.SYSTEM:
        raise ForbiddenException(
            message='Benchmark monitoring requires a user principal'
        )
    ctx.assert_org_role(OrgRole.OWNER)


def assert_benchmark_monitoring_visible(ctx, benchmark):
    assert_resource_visible(ctx, benchmark, not_found_message='Benchmark not found')
