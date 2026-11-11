from unittest.mock import patch

import pytest

from gpustack.schemas.models import DisaggregationSpec
from gpustack.utils.model_ports import model_port_requests
from gpustack.utils.sglang_port_layout import sglang_port_layout
from tests.utils.model import new_model


def _model(mode='sglang-mooncake', attention=True):
    return new_model(
        1,
        'sg',
        huggingface_repo_id='org/model',
        backend='SGLang',
        disaggregation=DisaggregationSpec(mode=mode),
        backend_parameters=['--tp=2']
        + (['--dp=2', '--enable-dp-attention'] if attention else []),
    )


@pytest.mark.parametrize('mode', ['sglang-mooncake', 'sglang-nixl'])
@pytest.mark.parametrize('managed', [False, True])
@pytest.mark.parametrize('attention', [False, True])
@pytest.mark.parametrize('allocation', [[2], [1, 1]])
def test_catalog_ports_are_combined_outside_native_sglang_layout(
    mode, managed, attention, allocation
):
    model = _model(mode, attention)
    for role in ('prefill', 'decode', 'router'):
        layout = sglang_port_layout(model, role, allocation, managed=managed)
        native = layout.port_requests()
        assert native[-1] == ('_api', 1, None)
        catalog = (
            [('bootstrap', 1, None)]
            if role == 'prefill'
            else [('prometheus', 1, None)] if role == 'router' else []
        )
        with patch.object(
            type(layout), 'port_requests', autospec=True, return_value=native
        ) as source:
            combined = model_port_requests(model, role, allocation, managed=managed)
        source.assert_called_once()
        assert len(source.call_args.args) == 1
        assert source.call_args.kwargs == {}
        assert combined == native[:-1] + catalog + native[-1:]


def test_catalog_band_sorting_is_owned_by_common_composition():
    model = _model()
    with patch(
        'gpustack.utils.model_ports.connector_port_requests',
        return_value=[('small', 1), ('wide', 4)],
    ):
        assert model_port_requests(model, 'prefill', [1, 1]) == [
            ('_sglang_connecting', 14, None),
            ('wide', 4, None),
            ('small', 1, None),
            ('_api', 1, None),
        ]
