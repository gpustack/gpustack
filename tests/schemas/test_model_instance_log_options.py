from gpustack.schemas.models import ServeLogOptionsResponse


def test_legacy_restart_counts_become_addressable_entries():
    """An old worker sends only restart_counts. Expanding them must fill in
    restart_count, or the pinned first-failure log stays unreachable: `previous`
    names just the second newest, and these lists are not contiguous."""
    response = ServeLogOptionsResponse.model_validate({"restart_counts": [199, 0, 198]})

    assert [(e.restart_count, e.previous) for e in response.restarts] == [
        (199, False),
        (198, True),
        (0, True),
    ]


def test_legacy_restart_counts_skip_unparsable_values():
    response = ServeLogOptionsResponse.model_validate(
        {"restart_counts": [2, "nope", None, 1]}
    )

    assert [e.restart_count for e in response.restarts] == [2, 1]


def test_restarts_from_a_current_worker_pass_through():
    """`restarts` present means the worker already reports restart_count."""
    response = ServeLogOptionsResponse.model_validate(
        {
            "restarts": [
                {"previous": False, "restart_count": 7},
                {"previous": True, "restart_count": 0},
            ]
        }
    )

    assert [(e.restart_count, e.previous) for e in response.restarts] == [
        (7, False),
        (0, True),
    ]


def test_missing_restart_counts_yields_no_entries():
    assert ServeLogOptionsResponse.model_validate({}).restarts == []
    assert (
        ServeLogOptionsResponse.model_validate({"restart_counts": None}).restarts == []
    )
