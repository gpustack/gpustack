"""Which Grafana dashboard a model's monitoring link lands on.

A disaggregated group and a plain deployment are not the same subject. The
model dashboard is not merely less specific for a group -- its request counters
double, because every request under PD traverses both roles -- so sending a
group there hands the reader a number that is 2x reality with nothing saying
so.
"""

from types import SimpleNamespace
import json
from pathlib import Path
import re
from unittest.mock import AsyncMock, patch

import pytest

from gpustack.api.exceptions import InternalServerErrorException
from gpustack.routes import models as route
from gpustack.schemas.models import DisaggregationSpec, Model, PDModeEnum


def _model(**kwargs) -> Model:
    return Model(
        id=kwargs.pop("id", 10),
        name=kwargs.pop("name", "m"),
        cluster_id=kwargs.pop("cluster_id", 1),
        **kwargs,
    )


def _cfg(**overrides):
    base = {
        "grafana_model_dashboard_uid": "gpustack-model",
        "grafana_pd_dashboard_uid": "gpustack-pd",
        "get_grafana_url": lambda: "http://grafana:3000",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


async def _redirect(model, cfg=None, mode=None):
    with (
        patch.object(route, "_get_model", AsyncMock(return_value=model)),
        patch.object(route, "get_global_config", lambda: cfg or _cfg()),
        patch.object(
            route.Cluster,
            "one_by_id",
            AsyncMock(return_value=SimpleNamespace(name="x1")),
        ),
        patch.object(
            route, "resolve_grafana_base_url", lambda *_: "http://grafana:3000"
        ),
        # Patched at its source, not on a route module: the dashboard link is
        # built in `utils.grafana` now, shared with the benchmark report, and
        # it reads the catalog through a deferred import.
        patch("gpustack.server.pd_mode_catalog.get_pd_mode", lambda _: mode),
    ):
        response = await route.get_model_dashboard(
            session=None, ctx=None, id=1, request=None
        )
    return response.headers["location"]


@pytest.mark.asyncio
async def test_a_plain_model_goes_to_the_model_dashboard():
    location = await _redirect(_model())
    assert "/d/gpustack-model/gpustack-model" in location
    assert "var-selected_model_id=10" in location
    assert "var-selected_cluster_id=1" in location
    # No role variable: the model dashboard has none, and a stray one would
    # ride into a saved URL that means nothing there.
    assert "counted_role" not in location


@pytest.mark.asyncio
async def test_a_disaggregated_group_goes_to_the_pd_dashboard():
    model = _model(disaggregation=DisaggregationSpec(mode=PDModeEnum.VLLM_NIXL))
    location = await _redirect(model)
    assert "/d/gpustack-pd/gpustack-pd" in location
    assert "var-model_name=m" in location
    assert "var-cluster_name=x1" in location


@pytest.mark.asyncio
async def test_the_counted_role_follows_the_connector():
    """The dashboard defaults to `decode`, so a pushing connector has to say so
    or its transfer panels read a flat zero."""
    model = _model(disaggregation=DisaggregationSpec(mode=PDModeEnum.SGLANG_MOONCAKE))
    pushes = SimpleNamespace(transfer_metrics=SimpleNamespace(read_from_role="prefill"))
    assert "var-counted_role=prefill" in await _redirect(model, mode=pushes)

    pulls = SimpleNamespace(transfer_metrics=SimpleNamespace(read_from_role="decode"))
    assert "var-counted_role=decode" in await _redirect(model, mode=pulls)


@pytest.mark.asyncio
async def test_an_unknown_mode_falls_back_to_decode():
    """A mode the catalog does not carry still gets a working link. Guessing
    wrong shows an empty panel, which beats no dashboard at all."""
    model = _model(disaggregation=DisaggregationSpec(mode=PDModeEnum.VLLM_NIXL))
    assert "var-counted_role=decode" in await _redirect(model, mode=None)


@pytest.mark.asyncio
async def test_a_group_is_refused_when_only_the_pd_uid_is_missing():
    """The uid that matters is the one this model will use, not the model
    dashboard's -- a deployment that configured only the model uid must not
    redirect a group to a dashboard that does not exist."""
    model = _model(disaggregation=DisaggregationSpec(mode=PDModeEnum.VLLM_NIXL))
    with pytest.raises(InternalServerErrorException):
        await _redirect(model, cfg=_cfg(grafana_pd_dashboard_uid=None))

    # The same deployment still serves a plain model.
    location = await _redirect(_model(), cfg=_cfg(grafana_pd_dashboard_uid=None))
    assert "/d/gpustack-model/" in location


def test_ascii_dashboard_digits_can_exceed_the_runtime_integer_limit():
    import sys
    from gpustack.routes.model_instances import _dashboard_query_target
    from gpustack.api.exceptions import BadRequestException

    limit = sys.get_int_max_str_digits()
    if not limit:
        pytest.skip('Runtime integer conversion limit is disabled')
    with pytest.raises(BadRequestException):
        _dashboard_query_target({'dp_rank': '9' * (limit + 1)})


def _dashboard():
    return json.loads(
        (
            Path(__file__).parents[2]
            / 'docker-compose/grafana/grafana_dashboards/gpustack-model.json'
        ).read_text()
    )


def _interpolate_dashboard_expression(expression, values):
    return (
        re.sub(
            r'\$\{(\w+):(raw|regex)\}',
            lambda m: '.*' if values[m[1]] == '$__all' else values[m[1]],
            expression,
        )
        .replace('$__rate_interval', '2m')
        .replace('$__range', '5m')
    )


@pytest.mark.parametrize(
    'value', ['.*', 'engine.[0]"\\💃', ' leading\ntrailing ', "x'y/$1"]
)
def test_fixed_dashboard_scope_quotes_literal_re2_without_changing_all(value):
    from gpustack.utils.grafana import model_dashboard_scope_params

    params = model_dashboard_scope_params(
        cluster_id=1,
        model_id=10,
        model_instance_id=100,
        restart_count=3,
        source_kind='vllm_engine',
        source_name=value,
        worker_id=2,
        dp_rank=4,
    )
    encoded = params['var-scope_source_name']
    assert json.loads(encoded).startswith('^')
    assert re.fullmatch(json.loads(encoded), value)
    assert not re.fullmatch(json.loads(encoded), value + 'other')
    assert '\\ud' not in encoded.lower()
    assert params['var-scope_endpoint_dp_rank'] == '"4"'
    assert params['var-scope_scheduler_dp_rank'] == '".*"'
    assert params['var-scope_legacy_enabled'] == '0'
    assert params['var-scope_bound'] == '1'
    defaults = model_dashboard_scope_params()
    assert defaults['var-scope_source_name'] == '".*"'
    assert encoded != defaults['var-scope_source_name']


def test_sglang_rank_scope_uses_native_dp_without_treating_attention_or_ep_as_dp():
    from gpustack.utils.grafana import model_dashboard_scope_params

    values = model_dashboard_scope_params(
        source_kind='sglang_engine', restart_count=3, dp_rank=2
    )
    assert values['var-scope_scheduler_dp_rank'] == '"2"'
    assert values['var-scope_endpoint_dp_rank'] == '".*"'
    assert not any('attn' in k or 'tp_rank' in k or 'ep_rank' in k for k in values)
    with pytest.raises(ValueError):
        model_dashboard_scope_params(source_kind='sglang_smg', dp_rank=2)


def test_real_template_has_fixed_scopes_and_scopes_every_business_expression():
    dashboard = _dashboard()
    variables = {v['name']: v for v in dashboard['templating']['list']}
    scopes = [name for name in variables if name.startswith('scope_')]
    assert len(scopes) >= 12
    assert all(
        variables[n]['type'] == 'textbox' and not variables[n].get('refresh')
        for n in scopes
    )
    context = {32, 33, 35, 28, 34, 30, 29}
    original_exprs = 0
    for panel in dashboard['panels']:
        for target in panel.get('targets', []):
            expr = target.get('expr', '')
            if not expr:
                continue
            if panel['id'] <= 53:
                original_exprs += 1
            assert '${scope_cluster_id:raw}' in expr
            assert '${scope_model_id:raw}' in expr
            if panel['id'] in context:
                assert 'model context' in panel['description'].lower()
                continue
            if panel['id'] <= 53:
                for name in (
                    'model_instance_id',
                    'restart_count',
                    'source_name',
                    'source_kind',
                    'worker_id',
                    'endpoint_dp_rank',
                    'scheduler_dp_rank',
                    'component_attempt',
                ):
                    assert '${scope_' + name + ':raw}' in expr
                assert 'gpustack:runtime_metrics_scrape_status' in expr
                assert 'gpustack:model_instance_source_info' in expr
                assert 'source_kind=""' in expr
                assert '${scope_legacy_enabled:raw}' in expr
                assert 'count_over_time(gpustack:runtime_metrics_scrape_status' in expr
                if panel['type'] == 'stat':
                    assert target['instant'] is True
                    assert target['range'] is False
                    assert panel['options']['reduceOptions']['calcs'] == ['last']
            assert 'or vector(0)' not in expr
            assert 'group_left' not in expr
            if 'histogram_quantile' in expr:
                assert 'sum by(le,' in expr or 'sum by (le,' in expr
    assert original_exprs == 75
    all_exprs = '\n'.join(
        t.get('expr', '') for p in dashboard['panels'] for t in p.get('targets', [])
    )
    for metric in (
        'gpustack:smg_router_requests_total',
        'gpustack:smg_worker_selections_total',
        'gpustack:smg_worker_requests_active',
        'gpustack:model_instance_placement_info',
    ):
        assert metric in all_exprs


@pytest.mark.asyncio
async def test_plain_model_dashboard_url_preselects_editable_ids_without_fixed_scope():
    from urllib.parse import parse_qs, urlsplit

    values = parse_qs(urlsplit(await _redirect(_model(id=10))).query)
    assert values['var-selected_model_id'] == ['10']
    assert values['var-selected_cluster_id'] == ['1']
    assert values['var-scope_bound'] == ['1']
    assert not any(
        k.startswith('var-scope_') and k != 'var-scope_bound' for k in values
    )
    assert values['var-model_name'] == ['m']
    assert values['var-cluster_name'] == ['x1']
    pd = parse_qs(
        urlsplit(
            await _redirect(
                _model(
                    id=10, disaggregation=DisaggregationSpec(mode=PDModeEnum.VLLM_NIXL)
                )
            )
        ).query
    )
    assert not any(k.startswith('var-scope_') for k in pd)


def test_model_dashboard_selectors_are_visible_cascading_and_scope_aware():
    dashboard = _dashboard()
    variables = {v['name']: v for v in dashboard['templating']['list']}
    for label in ('cluster', 'model', 'model_instance'):
        variable = variables[f'selected_{label}_id']
        assert variable['type'] == 'query'
        assert variable.get('hide', 0) == 0
        assert variable['datasource']['uid'] == '${DS_PROMETHEUS}'
        query = variable['query']['query']
        assert '${scope_' + label + '_id:raw}' in query
        if label != 'cluster':
            assert '${selected_cluster_id:regex}' in query
        if label == 'model_instance':
            assert '${selected_model_id:regex}' in query
        assert '(?<text>' in variable['regex']
        assert '(?<value>' in variable['regex']
        assert variable['includeAll']

    context = {32, 33, 35, 28, 34, 30, 29}
    for panel in dashboard['panels']:
        assert 'Model context:' not in panel['title']
        for target in panel.get('targets', []):
            expr = target.get('expr', '')
            if not expr:
                continue
            assert 'cluster_id=~"${selected_cluster_id:regex}"' in expr
            assert 'model_id=~"${selected_model_id:regex}"' in expr
            if panel['id'] not in context:
                assert (
                    'model_instance_id=~"${selected_model_instance_id:regex}"' in expr
                )


def test_fixed_instance_dashboard_preselects_ids_and_keeps_source_fences():
    from gpustack.utils.grafana import model_dashboard_scope_params

    params = model_dashboard_scope_params(
        cluster_id=1,
        model_id=10,
        model_instance_id=70,
        restart_count=3,
        source_kind='vllm_engine',
        dp_rank=2,
    )
    assert params['var-selected_cluster_id'] == '1'
    assert params['var-selected_model_id'] == '10'
    assert params['var-selected_model_instance_id'] == '70'
    assert params['var-scope_bound'] == '1'
    assert params['var-scope_restart_count'] == '"3"'
    assert params['var-scope_endpoint_dp_rank'] == '"2"'
    assert params['var-scope_legacy_enabled'] == '0'


@pytest.mark.parametrize(
    'selection, fixed_scope, expected',
    [
        ({'cluster': 1, 'model': 10}, {}, 5),
        ({'cluster': 1, 'model': 10, 'model_instance': 70}, {}, 2),
        ({'cluster': 1, 'model': 11}, {}, 7),
        ({'cluster': 2, 'model': 20}, {}, 11),
        ({'cluster': 1, 'model': 30}, {}, 13),
        ({'cluster': 2, 'model': 10}, {}, None),
        (
            {'cluster': 1, 'model': 10, 'model_instance': 70},
            dict(
                cluster_id=1,
                model_id=10,
                model_instance_id=70,
                restart_count=3,
                source_kind='vllm_engine',
                dp_rank=0,
            ),
            2,
        ),
        ({'cluster': 1, 'model': 11}, dict(model_id=10), None),
    ],
)
def test_model_dashboard_selection_filters_real_prometheus_queries(
    tmp_path, selection, fixed_scope, expected
):
    import os
    import shutil
    import subprocess
    import yaml
    from gpustack.utils.grafana import (
        build_model_dashboard_url,
        model_dashboard_scope_params,
    )
    from urllib.parse import parse_qs, urlsplit

    promtool = os.environ.get('PROMTOOL') or shutil.which('promtool')
    if not promtool:
        pytest.skip('Set PROMTOOL to validate dashboard queries with Prometheus')
    dashboard = _dashboard()
    values = {v['name']: v['current']['value'] for v in dashboard['templating']['list']}
    values.update(
        {f'selected_{key}_id': str(value) for key, value in selection.items()}
    )
    values.update(
        {
            k.removeprefix('var-'): v
            for k, v in model_dashboard_scope_params(**fixed_scope).items()
            if not k.startswith('var-selected_')
        }
    )
    if not fixed_scope:
        url = build_model_dashboard_url(_cfg(), 'http://grafana', _model(), 'x1')
        values.update(
            {
                k.removeprefix('var-'): v[0]
                for k, v in parse_qs(urlsplit(url).query).items()
                if not k.startswith('var-selected_')
            }
        )

    def interpolate(expr):
        return _interpolate_dashboard_expression(expr, values)

    rules = []
    for panel in dashboard['panels']:
        for target in panel.get('targets', []):
            if target.get('expr'):
                rules.append(
                    dict(
                        record=f"panel_{panel['id']}_{target['refId']}",
                        expr=interpolate(target['expr']),
                    )
                )
    for variable in dashboard['templating']['list']:
        if variable['type'] == 'query':
            rules.append(
                dict(
                    record=variable['name'],
                    expr=interpolate(
                        variable['query']['query'][len('query_result(') : -1]
                    ),
                )
            )
    rule_path = tmp_path / 'model-dashboard-rules.yml'
    rule_path.write_text(
        yaml.safe_dump({'groups': [dict(name='dashboard', rules=rules)]})
    )
    checked = subprocess.run(
        [promtool, 'check', 'rules', str(rule_path)], capture_output=True, text=True
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr

    inputs = []
    for cluster, model, instance, value, legacy in [
        (1, 10, 70, 2, False),
        (1, 10, 71, 3, False),
        (1, 11, 72, 7, False),
        (2, 20, 80, 11, False),
        (1, 30, 90, 13, True),
    ]:
        labels = dict(
            cluster_id=str(cluster),
            model_id=str(model),
            model_instance_id=str(instance),
        )
        if not legacy:
            labels.update(
                restart_count='3',
                source_kind='vllm_engine',
                source_name='engine',
                endpoint_dp_rank='0',
                worker_id='1',
            )

        def series(metric, extra=None):
            return (
                metric
                + '{'
                + ','.join(
                    k + '=' + json.dumps(v)
                    for k, v in {**labels, **(extra or {})}.items()
                )
                + '}'
            )

        inputs.append(
            dict(series=series('gpustack:num_requests_running'), values=f'{value}+0x5')
        )
        if not legacy:
            for metric, extra in [
                ('gpustack:runtime_metrics_scrape_status', {'status': 'success'}),
                ('gpustack:model_instance_source_info', {}),
            ]:
                inputs.append(dict(series=series(metric, extra), values='1+0x5'))
    expr = next(
        t['expr'] for p in dashboard['panels'] if p['id'] == 37 for t in p['targets']
    )
    path = tmp_path / 'model-dashboard-test.yml'
    path.write_text(
        yaml.safe_dump(
            dict(
                evaluation_interval='1m',
                tests=[
                    dict(
                        interval='1m',
                        input_series=inputs,
                        promql_expr_test=[
                            dict(
                                expr=interpolate(expr),
                                eval_time='5m',
                                exp_samples=(
                                    []
                                    if expected is None
                                    else [dict(labels='{}', value=expected)]
                                ),
                            )
                        ],
                    )
                ],
            )
        )
    )
    result = subprocess.run(
        [promtool, 'test', 'rules', str(path)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_compatibility_names_are_literal_patterns_separate_from_display_names():
    from gpustack.utils.grafana import model_dashboard_scope_params

    name = '引擎.[x]"\\path💃'
    params = model_dashboard_scope_params(
        model_id=10,
        model_name=name,
        cluster_name=name,
        model_instance_name=name,
    )
    for key in ('model_name', 'cluster_name', 'model_instance_name'):
        assert re.fullmatch(params['var-' + key], name)
        assert not re.fullmatch(params['var-' + key], name + 'other')
        assert params['var-display_' + key] == name


@pytest.mark.parametrize(
    'name',
    [
        None,
        'live5742-qwen-smoke',
        '.*',
        '引擎.[x]"\\path💃',
        'name`}|other_metric{a="b',
        ' leading\ntrailing ',
        "x'y/$1",
    ],
)
def test_template_name_matchers_remain_quoted_after_grafana_interpolation(name):
    from gpustack.utils.grafana import model_dashboard_scope_params

    dashboard = _dashboard()
    values = {
        variable['name']: variable['current']['value']
        for variable in dashboard['templating']['list']
    }
    names = ('cluster_name', 'model_name', 'model_instance_name')
    if name is not None:
        params = model_dashboard_scope_params(**dict.fromkeys(names, name))
        values.update(
            {key.removeprefix('var-'): value for key, value in params.items()}
        )

    def interpolate(match):
        key, formatter = match.groups()
        value = values[key]
        # Grafana Scenes 6.34 JSON formatting leaves string values unchanged.
        if formatter == 'raw' or isinstance(value, str):
            return str(value)
        assert formatter == 'json'
        return json.dumps(value, ensure_ascii=False)

    selector = r'\b(cluster_name|model_name|model_instance_name)=~'
    literal = r'(`[^`]*`|"(?:[^"\\]|\\.)*")'
    matched = 0
    for panel in dashboard['panels']:
        for target in panel.get('targets', []):
            expression = re.sub(
                r'\$\{([a-zA-Z0-9_]+):(raw|json)\}',
                interpolate,
                target.get('expr', ''),
            )
            selectors = re.findall(selector, expression)
            matchers = re.findall(selector + literal, expression)
            assert len(matchers) == len(selectors), expression
            for _, quoted in matchers:
                pattern = quoted[1:-1] if quoted.startswith('`') else json.loads(quoted)
                assert re.fullmatch(pattern, name if name is not None else 'any-model')
                if name is not None:
                    assert not re.fullmatch(pattern, name + 'other')
                matched += 1
    assert matched > 0


@pytest.mark.parametrize(
    'kwargs',
    [
        {'worker_id': -1},
        {'model_id': True},
        {'dp_rank': 1},
        {'source_kind': 'arbitrary_label'},
        {'source_kind': 'sglang_engine', 'component_attempt': 1},
    ],
)
def test_fixed_scope_helper_rejects_invalid_internal_binding(kwargs):
    from gpustack.utils.grafana import model_dashboard_scope_params

    with pytest.raises(ValueError):
        model_dashboard_scope_params(**kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'grafana_url, server_url, expected_server',
    [
        ('https://charts.example/observability', None, 'https://api.example/platform'),
        ('https://charts.example/grafana', None, 'https://api.example/platform'),
        (
            'https://charts.example/grafana',
            'https://public.example/gpu/',
            'https://public.example/gpu',
        ),
        (None, None, 'https://api.example/platform'),
        (None, 'https://public.example/gpu/', 'https://public.example/gpu'),
    ],
)
async def test_worker_link_uses_actual_server_request_root_not_grafana_origin(
    grafana_url, server_url, expected_server
):
    import httpx
    from fastapi import FastAPI, Request
    from urllib.parse import parse_qs, urlsplit

    app = FastAPI(root_path='/platform')

    @app.get('/models/{id}/dashboard')
    async def dashboard(id: int, request: Request):
        return await route.get_model_dashboard(None, None, id, request)

    cfg = _cfg(
        grafana_url=grafana_url,
        server_external_url=server_url,
        get_grafana_url=lambda: grafana_url or 'http://127.0.0.1:3000',
    )
    with (
        patch.object(route, '_get_model', AsyncMock(return_value=_model(id=10))),
        patch.object(
            route.Cluster,
            'one_by_id',
            AsyncMock(return_value=SimpleNamespace(name='cluster')),
        ),
        patch.object(route, 'get_global_config', lambda: cfg),
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url='https://api.example'
        ) as client:
            response = await client.get('/models/10/dashboard')
    assert response.status_code == 302
    location = response.headers['location']
    values = parse_qs(urlsplit(location).query)
    assert values['var-gpustack_base_url'] == [expected_server]
    placement = next(p for p in _dashboard()['panels'] if p['id'] == 66)
    link = placement['fieldConfig']['overrides'][0]['properties'][0]['value'][0]['url']
    actual_worker_url = link.replace(
        '${gpustack_base_url:raw}', values['var-gpustack_base_url'][0]
    ).replace('${__value.raw}', '2')
    assert actual_worker_url == expected_server + '/v2/workers/2/dashboard'
    assert location.startswith((grafana_url or expected_server + '/grafana') + '/d/')


@pytest.mark.parametrize(
    'server_url, expected',
    [
        (None, None),
        ('https://public.example/gpu/', 'https://public.example/gpu'),
    ],
)
def test_pure_dashboard_helper_uses_only_configured_server_without_request(
    server_url, expected
):
    from gpustack.utils.grafana import build_model_dashboard_url
    from urllib.parse import parse_qs, urlsplit

    location = build_model_dashboard_url(
        _cfg(server_external_url=server_url),
        'https://external.example/grafana',
        _model(id=10),
    )
    params = parse_qs(urlsplit(location).query)
    if expected is None:
        assert 'var-gpustack_base_url' not in params
    else:
        assert params['var-gpustack_base_url'] == [expected]


def _render_dashboard_legend(panel, target, labels, prefix=''):
    name = re.sub(
        r'\{\{\s*(.+?)\s*\}\}',
        lambda m: labels.get(m[1]) or m[1],
        target['legendFormat'].replace('${legend_instance_prefix:raw}', prefix),
    )
    for transform in panel.get('transformations', []):
        if transform['id'] != 'renameByRegex':
            continue
        options = transform['options']
        pattern = options['regex']
        global_ = pattern.startswith('/') and pattern.endswith('/g')
        if pattern.startswith('/'):
            pattern = pattern[1:-2] if global_ else pattern[1:-1]
        else:
            pattern = '^' + pattern + '$'
        name = re.sub(
            pattern,
            lambda match: re.sub(
                r'\$(\d+)',
                lambda group: match[int(group[1])] or '',
                options['renamePattern'],
            ),
            name,
            count=0 if global_ else 1,
        )
    return name


@pytest.mark.parametrize(
    'panel_id,ref,labels,expected',
    [
        (
            22,
            'A',
            {'model_instance_name': 'instance-a', 'endpoint_dp_rank': '0'},
            '[instance-a] · DP 0',
        ),
        (
            9,
            'B',
            {
                'model_instance_name': 'instance-b',
                'endpoint_dp_rank': '1',
                'restart_count': '2',
            },
            'P99 · [instance-b] · DP 1 · restart 2',
        ),
        (
            9,
            'B',
            {
                'model_instance_name': 'instance-a',
                'endpoint_dp_rank': '0',
                'dp_rank': '0',
            },
            'P99 · [instance-a] · DP 0',
        ),
        (
            22,
            'A',
            {'model_instance_name': 'instance-a', 'dp_rank': '1'},
            '[instance-a] · logical DP 1',
        ),
        (
            22,
            'A',
            {
                'model_instance_name': 'instance-a',
                'endpoint_dp_rank': '0',
                'dp_rank': '1',
            },
            '[instance-a] · DP 0 · logical DP 1',
        ),
        (41, 'D', {}, 'Total'),
        (
            9,
            'B',
            {
                'model_instance_name': 'a.[x] · DP · restart · worker',
                'restart_count': '0',
            },
            'P99 · [a.[x] · DP · restart · worker] · restart 0',
        ),
        (4, 'A', {'model_instance_name': 'instance-a'}, '[instance-a]'),
        (
            65,
            'A',
            {
                'model_instance_name': 'instance-a',
                'worker': 'worker',
                'restart_count': '2',
                'component_attempt': '3',
            },
            '[instance-a] · SMG · worker · restart 2 · attempt 3',
        ),
    ],
)
@pytest.mark.parametrize('fixed_instance', [False, True])
def test_dashboard_legends_keep_instance_names_in_browsable_and_fixed_views(
    panel_id, ref, labels, expected, fixed_instance
):
    from gpustack.utils.grafana import model_dashboard_scope_params

    panel = next(p for p in _dashboard()['panels'] if p['id'] == panel_id)
    target = next(t for t in panel['targets'] if t['refId'] == ref)
    params = model_dashboard_scope_params(
        model_instance_id=70 if fixed_instance else None
    )
    assert (
        _render_dashboard_legend(
            panel, target, labels, params['var-legend_instance_prefix']
        )
        == expected
    )


def test_instance_legends_distinguish_restarts_without_changing_names():
    dashboard = _dashboard()
    for panel in dashboard['panels']:
        if panel['type'] != 'timeseries':
            continue
        for target in panel.get('targets', []):
            if 'model_instance_name' not in target.get('legendFormat', ''):
                assert '${legend_instance_prefix' not in target.get('legendFormat', '')
                continue
            labels = dict(
                model_instance_name='engine.[0] · DP',
                endpoint_dp_rank='0',
                dp_rank='0',
                endpoint='/generate',
                streaming='false',
                policy='cache_aware',
                worker='worker',
                component_attempt='0',
            )
            names = [
                _render_dashboard_legend(
                    panel, target, {**labels, 'restart_count': str(n)}
                )
                for n in (0, 1)
            ]
            assert len(set(names)) == 2, (panel['id'], target['refId'], names)
            assert all('engine.[0] · DP' in n for n in names)
        mappings = panel['fieldConfig']['defaults']['mappings']
        assert any(
            m['type'] == 'special'
            and m['options']['match'] == 'null+nan'
            and m['options']['result']['text'] == 'No data'
            for m in mappings
        )
    for panel in dashboard['panels']:
        if panel['type'] == 'heatmap':
            assert all(t['legendFormat'] == '{{le}}' for t in panel['targets'])
    mfu = next(p for p in dashboard['panels'] if p['id'] == 52)
    assert mfu['fieldConfig']['defaults']['unit'] == 'suffix:TFLOP/s/GPU'


@pytest.mark.parametrize(
    'kind', ['legacy', 'vllm_engine', 'vllm_hybrid', 'sglang_engine']
)
def test_aggregated_dashboard_queries_return_instance_names(tmp_path, kind):
    import os
    import shutil
    import subprocess
    import yaml

    promtool = os.environ.get('PROMTOOL') or shutil.which('promtool')
    if not promtool:
        pytest.skip('Set PROMTOOL to validate dashboard queries with Prometheus')
    dashboard = _dashboard()
    values = {v['name']: v['current']['value'] for v in dashboard['templating']['list']}
    targets = [
        t
        for p in dashboard['panels']
        for t in p.get('targets', [])
        if re.search(
            r'(?:sum|avg) by\s*\([^)]*\bmodel_instance_id\b', t.get('expr', '')
        )
    ]
    assert len(targets) == 33
    metrics = {
        metric
        for t in targets
        for metric in re.findall(r'gpustack:[\w:]+', t['expr'])
        if metric
        not in (
            'gpustack:runtime_metrics_scrape_status',
            'gpustack:model_instance_source_info',
        )
    }
    inputs = {}

    def add(metric, labels, slope=None):
        series = (
            metric
            + '{'
            + ','.join(k + '=' + json.dumps(v) for k, v in sorted(labels.items()))
            + '}'
        )
        inputs[series] = {
            'series': series,
            'values': '1+0x5' if slope is None else f'0+{slope}x5',
        }

    names = ('alpha.[x] · DP', 'beta')
    for instance_id, name in zip((70, 71), names):
        for rank in [None] if kind == 'legacy' else [0, 1]:
            labels = dict(
                cluster_id='1',
                model_id='10',
                model_instance_id=str(instance_id),
                model_instance_name=name,
            )
            if kind != 'legacy':
                labels.update(
                    restart_count='3',
                    source_kind=kind,
                    worker_id='1',
                    source_name=f'engine-{rank}' if kind == 'vllm_engine' else kind,
                )
                source_labels = dict(labels)
                if kind == 'vllm_engine':
                    labels['endpoint_dp_rank'] = source_labels['endpoint_dp_rank'] = (
                        str(rank)
                    )
                else:
                    labels['dp_rank'] = str(rank)
                add('gpustack:model_instance_source_info', source_labels)
                add(
                    'gpustack:runtime_metrics_scrape_status',
                    {**source_labels, 'status': 'success'},
                )
            for metric in metrics:
                if metric.endswith('_bucket'):
                    for bound, slope in [('1', 10), ('2', 20), ('+Inf', 20)]:
                        add(metric, {**labels, 'le': bound}, slope)
                else:
                    extra = (
                        {'finished_reason': 'stop'}
                        if 'request_finished' in metric
                        else {}
                    )
                    add(metric, {**labels, **extra}, 600)
    expected = [
        dict(
            labels='{model_instance_name=' + json.dumps(n) + '}',
            value=1 if kind == 'legacy' else 2,
        )
        for n in names
    ]
    checks = [
        dict(
            expr='count by (model_instance_name) ('
            + _interpolate_dashboard_expression(t['expr'], values)
            + ')',
            eval_time='5m',
            exp_samples=expected,
        )
        for t in targets
    ]
    path = tmp_path / 'instance-name-queries.yml'
    path.write_text(
        yaml.safe_dump(
            dict(
                evaluation_interval='1m',
                tests=[
                    dict(
                        interval='1m',
                        input_series=list(inputs.values()),
                        promql_expr_test=checks,
                    )
                ],
            )
        )
    )
    result = subprocess.run(
        [promtool, 'test', 'rules', str(path)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    'instance_id, expected',
    [(None, '[{{model_instance_name}}] · [{{model_instance_id}}] · '), (70, '')],
)
def test_fixed_instance_scope_sets_native_legend_prefix(instance_id, expected):
    from gpustack.utils.grafana import model_dashboard_scope_params

    params = model_dashboard_scope_params(model_instance_id=instance_id)
    assert params['var-legend_instance_prefix'] == expected


@pytest.mark.asyncio
async def test_pd_dashboard_names_cannot_break_quoted_promql():
    from urllib.parse import parse_qs, urlsplit
    from gpustack.utils.grafana import build_model_dashboard_url

    name = 'm.[x]"\\path\nnext'
    model = _model(
        name=name, disaggregation=DisaggregationSpec(mode=PDModeEnum.VLLM_NIXL)
    )
    url = build_model_dashboard_url(_cfg(), 'http://grafana', model, cluster_name=name)
    params = parse_qs(urlsplit(url).query)
    for field in ('model_name', 'cluster_name'):
        assert json.loads('"' + params['var-' + field][0] + '"') == name


def test_manifest_dashboard_legends_use_exported_labels():
    panels = {p['id']: p for p in _dashboard()['panels']}
    for id_ in (61, 62, 66):
        for target in panels[id_]['targets']:
            worker_label = (
                '{{placement_worker_name}}' if id_ == 66 else '{{worker_name}}'
            )
            assert worker_label in target['legendFormat']
            assert '{{worker}}' not in target['legendFormat']
            assert '{{policy}}' not in target['legendFormat']
    assert '{{status}}' in panels[61]['targets'][0]['legendFormat']


@pytest.mark.parametrize('worker_id', [1, 3])
def test_hybrid_dashboard_binds_shared_api_and_never_synthetic_rank(worker_id):
    from gpustack.routes.model_instances import (
        _dashboard_query_target,
        _dashboard_source_scope,
    )
    from gpustack.utils.grafana import model_dashboard_scope_params
    from tests.worker.test_runtime_metrics import _managed_instance, _hybrid_tasks
    from gpustack.api.exceptions import NotFoundException

    instance = _managed_instance(_hybrid_tasks())
    raw = dict(component='vllm_hybrid', worker_id=str(worker_id), restart_count='3')
    scope = _dashboard_source_scope(instance, _dashboard_query_target(raw))
    params = model_dashboard_scope_params(**scope)
    assert params['var-scope_source_kind'] == '"^vllm_hybrid$"'
    assert params['var-scope_worker_id'] == f'"{worker_id}"'
    assert params['var-scope_endpoint_dp_rank'] == '".*"'
    assert params['var-scope_scheduler_dp_rank'] == '".*"'
    for changes in ({'worker_id': '2'}, {'worker_id': '9'}, {'restart_count': '2'}):
        with pytest.raises(NotFoundException):
            _dashboard_source_scope(
                instance, _dashboard_query_target({**raw, **changes})
            )


@pytest.mark.parametrize('filters', [{'dp_rank': '0'}, {'attempt': '0'}])
def test_hybrid_dashboard_rejects_rank_and_smg_attempt(filters):
    from gpustack.routes.model_instances import _dashboard_query_target
    from gpustack.api.exceptions import BadRequestException
    from gpustack.utils.grafana import model_dashboard_scope_params

    with pytest.raises(BadRequestException):
        _dashboard_query_target(dict(component='vllm_hybrid', **filters))
    with pytest.raises(ValueError):
        model_dashboard_scope_params(
            source_kind='vllm_hybrid',
            **{
                'dp_rank' if 'dp_rank' in filters else 'component_attempt': 0,
            },
        )


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['deleted', 'cluster', 'missing', 'identity'])
async def test_hybrid_dashboard_refuses_changed_api_owner(monkeypatch, change):
    from gpustack.routes import model_instances as instances_route
    from gpustack.api.exceptions import NotFoundException
    from tests.routes.test_model_instance_log_download import _hybrid_log_instance

    worker = SimpleNamespace(id=3, cluster_id=1, deleted_at=None)
    if change == 'deleted':
        worker.deleted_at = 'deleted'
    elif change == 'cluster':
        worker.cluster_id = 2
    elif change == 'identity':
        worker.id = 4
    else:
        worker = None
    monkeypatch.setattr(
        instances_route.Worker, 'one_by_id', AsyncMock(return_value=worker)
    )
    with pytest.raises(NotFoundException):
        await instances_route._validate_dashboard_owner(None, _hybrid_log_instance(), 3)


@pytest.mark.asyncio
async def test_hybrid_dashboard_checks_tenant_before_resolving_listener(monkeypatch):
    from gpustack.routes import model_instances as instances_route
    from gpustack.api.exceptions import NotFoundException
    from tests.routes.test_model_instance_log_download import _hybrid_log_instance
    from unittest.mock import Mock

    monkeypatch.setattr(
        instances_route.ModelInstance,
        'one_by_id',
        AsyncMock(return_value=_hybrid_log_instance()),
    )
    monkeypatch.setattr(
        instances_route,
        'assert_resource_visible',
        Mock(side_effect=NotFoundException(message='Model instance not found')),
    )
    resolve = Mock()
    monkeypatch.setattr(instances_route, '_dashboard_source_scope', resolve)
    with pytest.raises(NotFoundException):
        await instances_route.get_model_instance_dashboard(
            None, None, 7, None, component='vllm_hybrid', worker_id='3'
        )
    resolve.assert_not_called()


@pytest.mark.asyncio
async def test_pd_dashboard_propagates_server_base_for_instance_links():
    from urllib.parse import parse_qs, urlsplit
    from gpustack.utils.grafana import build_model_dashboard_url

    location = build_model_dashboard_url(
        _cfg(server_external_url='https://gpustack.example/prefix'),
        'https://grafana.example',
        _model(disaggregation=DisaggregationSpec(mode=PDModeEnum.VLLM_NIXL)),
    )
    assert parse_qs(urlsplit(location).query)['var-gpustack_base_url'] == [
        'https://gpustack.example/prefix'
    ]


@pytest.mark.parametrize(
    'panel_id,selector', [(58, 'dp_rank'), (59, 'component=vllm_hybrid')]
)
def test_pd_endpoint_links_preserve_identity_and_shared_api_semantics(
    panel_id, selector
):
    dashboard = json.loads(
        (
            Path(__file__).parents[2]
            / 'docker-compose/grafana/grafana_dashboards/gpustack-pd.json'
        ).read_text()
    )
    panel = next(p for p in dashboard['panels'] if p['id'] == panel_id)
    assert (
        'source_kind="vllm_engine"' in panel['targets'][0]['expr']
        if panel_id == 58
        else 'source_kind="vllm_hybrid"' in panel['targets'][0]['expr']
    )
    properties = panel['fieldConfig']['overrides'][0]['properties']
    url = next(p['value'][0]['url'] for p in properties if p['id'] == 'links')
    assert '/v2/model-instances/${__data.fields.model_instance_id}/dashboard' in url
    assert 'restart_count=${__data.fields.restart_count}' in url
    assert 'worker_id=${__data.fields.worker_id}' in url
    assert selector in url
    assert ('dp_rank=' in url) == (panel_id == 58)
    assert panel['targets'][0]['instant'] and not panel['targets'][0]['range']
    assert panel['transformations'][0]['options']['renameByName'] == {}
    for identity in ('model_instance_id', 'restart_count'):
        assert not panel['transformations'][0]['options']['excludeByName'].get(identity)
        override = next(
            item
            for item in panel['fieldConfig']['overrides']
            if item['matcher'] == {'id': 'byName', 'options': identity}
        )
        assert {'id': 'custom.hidden', 'value': True} in override['properties']


@pytest.mark.parametrize('kind', ['vllm_engine', 'vllm_hybrid'])
@pytest.mark.parametrize('rank_count', [1, 2, 4])
def test_pd_queries_aggregate_normalized_rank_metrics_once(tmp_path, kind, rank_count):
    import os
    import shutil
    import subprocess
    import yaml
    from prometheus_client.exposition import CONTENT_TYPE_LATEST
    from tests.worker.test_runtime_metrics import (
        _source_aggregator,
        _managed_instance,
        _families,
    )
    from gpustack.schemas.models import BackendEnum

    promtool = os.environ.get('PROMTOOL') or shutil.which('promtool')
    if not promtool:
        pytest.skip('Set PROMTOOL to validate dashboard queries with Prometheus')
    aggregator, _ = _source_aggregator(_managed_instance())
    unified = {}
    local_size = 1 if kind == 'vllm_engine' else min(2, rank_count)
    for role in ('prefill', 'decode'):
        for first in range(0, rank_count, local_size):
            ranks = (
                range(rank_count)
                if kind == 'vllm_engine'
                else range(first, first + local_size)
            )
            definitions = {
                'prompt_tokens_total': ('counter', [('', 60)]),
                'generation_tokens_total': ('counter', [('', 120)]),
                'e2e_request_latency_seconds': (
                    'histogram',
                    [
                        ('_bucket', 30, '0.1'),
                        ('_bucket', 60, '1'),
                        ('_bucket', 60, '+Inf'),
                        ('_sum', 30),
                        ('_count', 60),
                    ],
                ),
                'nixl_xfer_time_seconds': (
                    'histogram',
                    [
                        ('_bucket', 30, '0.1'),
                        ('_bucket', 60, '1'),
                        ('_bucket', 60, '+Inf'),
                        ('_sum', 30),
                        ('_count', 60),
                    ],
                ),
                'nixl_bytes_transferred': (
                    'summary',
                    [('_sum', 61440), ('_count', 60)],
                ),
                'kv_cache_usage_perc': ('gauge', [('', 0.25)]),
            }
            text = []
            for name, (type_, samples) in definitions.items():
                text.append(f'# TYPE vllm:{name} {type_}')
                for rank in ranks:
                    engine = rank if kind == 'vllm_engine' else rank - first
                    for suffix, value, *le in samples:
                        labels = f'engine="{engine}"' + (f',le="{le[0]}"' if le else '')
                        value = value + rank * 0.1 if type_ == 'gauge' else value
                        text.append(f'vllm:{name}{suffix}{{{labels}}} {value}')
            labels = dict(
                cluster_name='x1',
                model_name='m',
                role=role,
                model_instance_name=f'{role}-instance',
                worker_id=str(first + 1),
                source_kind=kind,
            )
            if kind == 'vllm_engine':
                labels['endpoint_dp_rank'] = str(first)
            aggregator._process_endpoint_metrics(
                _families('\n'.join(text) + '\n', CONTENT_TYPE_LATEST),
                labels,
                BackendEnum.VLLM,
                '0.19.0',
                unified,
                {},
                aggregator._metrics_config_cache['config'],
            )
    inputs = []
    for family in unified.values():
        for sample in family.samples:
            labels = ','.join(
                f'{k}={json.dumps(v)}' for k, v in sorted(sample.labels.items())
            )
            inputs.append(
                {
                    'series': sample.name + '{' + labels + '}',
                    'values': (
                        f'{sample.value}x5'
                        if family.type == 'gauge'
                        else f'0+{sample.value}x5'
                    ),
                }
            )
    dashboard = json.loads(
        (
            Path(__file__).parents[2]
            / 'docker-compose/grafana/grafana_dashboards/gpustack-pd.json'
        ).read_text()
    )
    checks = []
    for panel_id, ref, expected in [
        (44, 'A', rank_count),
        (44, 'C', rank_count * 2),
        (45, 'D', 0.5),
        (14, 'A', 0.5),
        (15, 'A', 1024),
        (20, 'B', 0.1 + (1 - 0.1) * (0.95 - 0.5) / 0.5),
    ]:
        panel = next(p for p in dashboard['panels'] if p['id'] == panel_id)
        expr = next(t['expr'] for t in panel['targets'] if t['refId'] == ref)
        for key, value in [
            ('cluster_name', 'x1'),
            ('model_name', 'm'),
            ('counted_role', 'decode'),
            ('role', 'prefill|decode'),
            ('__rate_interval', '2m'),
            ('__range', '5m'),
        ]:
            expr = expr.replace('$' + key, value)
        checks.append(
            {
                'expr': expr,
                'eval_time': '5m',
                'exp_samples': [{'labels': '{}', 'value': expected}],
            }
        )
    kv = next(p for p in dashboard['panels'] if p['id'] == 49)['targets'][0]['expr']
    for key, value in [
        ('cluster_name', 'x1'),
        ('model_name', 'm'),
        ('role', 'prefill|decode'),
    ]:
        kv = kv.replace('$' + key, value)
    checks.append(
        {
            'expr': kv,
            'eval_time': '5m',
            'exp_samples': [
                {
                    'labels': '{role="'
                    + role
                    + '",model_instance_name="'
                    + role
                    + '-instance"}',
                    'value': 0.25 + (rank_count - 1) * 0.1,
                }
                for role in ('prefill', 'decode')
            ],
        }
    )
    case = {
        'evaluation_interval': '1m',
        'tests': [
            {'interval': '1m', 'input_series': inputs, 'promql_expr_test': checks}
        ],
    }
    path = tmp_path / 'pd-rank-queries.yml'
    path.write_text(yaml.safe_dump(case))
    result = subprocess.run(
        [promtool, 'test', 'rules', str(path)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_all_pd_dashboard_expressions_parse_in_prometheus(tmp_path):
    import os
    import shutil
    import subprocess
    import yaml

    promtool = os.environ.get('PROMTOOL') or shutil.which('promtool')
    if not promtool:
        pytest.skip('Set PROMTOOL to validate dashboard queries with Prometheus')
    dashboard = json.loads(
        (
            Path(__file__).parents[2]
            / 'docker-compose/grafana/grafana_dashboards/gpustack-pd.json'
        ).read_text()
    )
    rules = []
    for panel in dashboard['panels']:
        for target in panel.get('targets', []):
            expr = target.get('expr')
            if not expr:
                continue
            for key, value in [
                ('cluster_name', 'x1'),
                ('model_name', 'm'),
                ('counted_role', 'decode'),
                ('role', 'prefill|decode'),
                ('__rate_interval', '2m'),
                ('__range', '5m'),
            ]:
                expr = expr.replace('$' + key, value)
            rules.append(
                {'record': f"pd_panel_{panel['id']}_{target['refId']}", 'expr': expr}
            )
    path = tmp_path / 'pd-dashboard-rules.yml'
    path.write_text(
        yaml.safe_dump({'groups': [{'name': 'pd-dashboard', 'rules': rules}]})
    )
    result = subprocess.run(
        [promtool, 'check', 'rules', str(path)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
