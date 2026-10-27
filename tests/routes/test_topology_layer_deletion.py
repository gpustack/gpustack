"""Deleting a topology layer a model still gathers on.

Saving the whole chain is the only delete there is — a layer is gone when the
next `PUT /clusters/{id}` arrives without it — so this guard lives on the
update path rather than behind a delete endpoint that does not exist.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gpustack.api.exceptions import BadRequestException
from gpustack.routes.clusters import check_topology_layers_not_stranded
from gpustack.schemas.clusters import ClusterTopology, ClusterUpdate
from gpustack.topology.tree import NODE_LAYER
from tests.utils.topology_layers import layer_dict, lid

POD = "dc/pod"


def _model(name, layer=None):
    return SimpleNamespace(
        name=name,
        gather=SimpleNamespace(layer=layer) if layer else None,
    )


async def _check(models, layers):
    """Run the guard over a save carrying `layers`."""
    cluster = SimpleNamespace(id=1)
    update = ClusterUpdate(
        name="c1", topology=ClusterTopology.model_validate({"layers": layers})
    )
    with patch(
        "gpustack.schemas.models.Model.all_by_field",
        return_value=models,
    ):
        await check_topology_layers_not_stranded(None, cluster, update)


@pytest.mark.asyncio
async def test_dropping_a_custom_layer_nobody_gathers_on_is_fine():
    await _check([_model("plain")], [])


@pytest.mark.asyncio
async def test_keeping_the_layer_is_fine():
    layers = [layer_dict("Pod", [POD], parent="rack")]
    await _check([_model("pd-a", lid("Pod"))], layers)


@pytest.mark.asyncio
async def test_dropping_a_layer_a_model_gathers_on_is_refused():
    """Refusing is the only honest option. Clearing the models' requirement
    rewrites what someone asked for; leaving them pointing at nothing is
    worse, because the solver stands an unresolvable gather down rather than
    failing — a `MustGather` would go on being stored while enforcing
    nothing."""
    with pytest.raises(BadRequestException) as e:
        await _check(
            [_model("pd-a", lid("Pod")), _model("pd-b", lid("Pod"))],
            [],
        )
    # Names the models, because "something references it" leaves the operator
    # with a fleet to search.
    assert "pd-a, pd-b" in str(e.value.message)


@pytest.mark.asyncio
async def test_switching_off_a_rung_a_model_gathers_on_is_refused_too():
    """The second way a tier disappears, and the easy one to miss: the row
    survives, so it does not look like a removal — but `active()` drops it,
    so the model's `MustGather` would name a rung the solver never groups
    by."""
    with pytest.raises(BadRequestException) as e:
        await _check([_model("pd-a", lid("rack"))], [layer_dict("rack", disabled=True)])
    assert "pd-a" in str(e.value.message)


@pytest.mark.asyncio
async def test_an_untouched_builtin_rung_is_never_stranded():
    await _check([_model("pd-a", lid("rack"))], [])
    await _check([_model("pd-a", NODE_LAYER)], [])


@pytest.mark.asyncio
async def test_a_save_that_does_not_mention_topology_is_not_a_delete():
    """`ClusterUpdate` carries every field; a PUT that only renames the
    cluster must not read as "and remove all the layers"."""
    cluster = SimpleNamespace(id=1)
    update = ClusterUpdate(name="c1")
    with patch(
        "gpustack.schemas.models.Model.all_by_field",
        return_value=[_model("pd-a", lid("Pod"))],
    ):
        await check_topology_layers_not_stranded(None, cluster, update)
