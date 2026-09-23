"""Port allocation: contiguous bands and the occupancy probe underneath them.

The band allocator exists because a connector can derive several ports from
one base, and the probe binds rather than connects because a connect can only
see a port that is *already listening on the advertised address* — the two are
the same story from opposite ends, so they are pinned in one file.
"""

import socket

import pytest

from gpustack.utils import network
from gpustack.utils.network import (
    PortRangeExhaustedError,
    get_free_band,
    is_port_available,
)


@pytest.fixture
def all_ports_free(monkeypatch):
    """Take the real probe out of the loop.

    The allocation tests are about the scan, not about the machine they run
    on; probing real ports would make them depend on whatever else the host
    happens to be running.
    """
    monkeypatch.setattr(network, "is_port_available", lambda port, host: True)


@pytest.fixture
def ports_busy(monkeypatch):
    """Probe answers driven by a set the test controls."""

    def install(busy):
        monkeypatch.setattr(
            network, "is_port_available", lambda port, host: port not in busy
        )

    return install


def test_get_free_band_returns_the_first_contiguous_run(all_ports_free):
    assert get_free_band((40000, 40063), count=4) == 40000


def test_get_free_band_is_first_fit_not_random(all_ports_free):
    """Two consecutive allocations must sit next to each other.

    This is the whole reason the allocator is not `get_free_port`
    generalized: random picks leave one-port holes all over a 64-port pool.
    """
    taken = set()
    first = get_free_band((40000, 40063), count=2, unavailable_ports=taken)
    taken |= {first, first + 1}
    second = get_free_band((40000, 40063), count=2, unavailable_ports=taken)

    assert (first, second) == (40000, 40002)


def test_get_free_band_skips_past_the_blocker_not_one_port(all_ports_free):
    """A band starting anywhere at or before the blocker cannot fit, so the
    scan resumes after it."""
    taken = {40001}
    assert get_free_band((40000, 40063), count=2, unavailable_ports=taken) == 40002


def test_get_free_band_keeps_the_pool_defragmented(all_ports_free):
    """Eight single-port allocations must still leave a run of eight.

    Random allocation cannot promise this: eight random picks in a 64-port
    pool statistically shatter it, and the ninth request — the TP8 Ascend
    band — fails while the pool is still 87% empty.
    """
    taken = set()
    for _ in range(8):
        taken.add(get_free_band((40000, 40063), count=1, unavailable_ports=taken))

    assert get_free_band((40000, 40063), count=8, unavailable_ports=taken) == 40008


def test_get_free_band_count_one_is_a_single_free_port(all_ports_free):
    taken = {40000, 40001}
    assert get_free_band((40000, 40063), count=1, unavailable_ports=taken) == 40002


def test_get_free_band_finds_the_only_gap_that_fits(ports_busy):
    """Fragmented pool: plenty free, exactly one usable run."""
    # Everything busy except 40010-40012.
    ports_busy({p for p in range(40000, 40064) if p not in (40010, 40011, 40012)})

    assert get_free_band((40000, 40063), count=3) == 40010


def test_get_free_band_fragmented_pool_has_no_wide_gap(ports_busy):
    """Every other port free: 32 free ports, no run of two."""
    ports_busy({p for p in range(40000, 40064) if p % 2 == 0})

    with pytest.raises(PortRangeExhaustedError) as excinfo:
        get_free_band((40000, 40063), count=2)

    message = str(excinfo.value)
    assert "no gap of 2 consecutive ports" in message
    # The distinction matters: "32 free but none adjacent" is a different
    # operator action from "the pool is full".
    assert "32 of the 64 ports" in message


def test_get_free_band_exhaustion_message_carries_the_arithmetic(all_ports_free):
    taken = set(range(40000, 40060))

    with pytest.raises(PortRangeExhaustedError) as excinfo:
        get_free_band((40000, 40063), count=8, unavailable_ports=taken)

    message = str(excinfo.value)
    assert "8 consecutive free port(s)" in message  # how many were wanted
    assert "only 4 of the 64 ports" in message  # what is left
    assert "at least 72 ports" in message  # what to do about it
    assert "40000-40071" in message


def test_get_free_band_rejects_a_band_wider_than_the_range(all_ports_free):
    with pytest.raises(PortRangeExhaustedError):
        get_free_band((40000, 40007), count=9)


def test_get_free_band_rejects_a_nonsense_count(all_ports_free):
    with pytest.raises(ValueError):
        get_free_band((40000, 40063), count=0)


def test_get_free_band_memoizes_probed_occupancy(ports_busy):
    """A port found busy by probing is written back into the caller's set,
    the way `get_free_port` does, so the next band in the same allocation
    pass doesn't pay for the probe again."""
    ports_busy({40000})
    taken = set()

    assert get_free_band((40000, 40063), count=1, unavailable_ports=taken) == 40001
    assert 40000 in taken


def test_get_free_band_accepts_the_config_spelling_of_a_range(all_ports_free):
    """`service_port_range` is carried as a string everywhere in config."""
    assert get_free_band("40000-40063", count=2) == 40000


# --- the probe itself -------------------------------------------------------


def test_is_port_available_sees_a_bound_but_unlistened_socket():
    """The case a connect probe is blind to.

    A socket that has bound but not yet called listen() refuses connections
    exactly like a free port does, so a connect probe hands the port out and
    the collision surfaces later as an engine wedged in `starting`.
    """
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        holder.bind(("127.0.0.1", 0))
        port = holder.getsockname()[1]
        # No listen() on purpose.

        # What a connect probe sees: nothing to connect to.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.5)
            assert probe.connect_ex(("127.0.0.1", port)) != 0

        # What the bind probe sees.
        assert is_port_available(port, "127.0.0.1") is False
    finally:
        holder.close()


def test_is_port_available_sees_a_listener_on_another_local_address():
    """Bound to loopback, probed on a different address.

    A connect probe only ever reaches the address it was given, so a
    server on 127.0.0.1 is invisible when the allocator probes the worker's
    advertised IP. Binding 0.0.0.0 conflicts with any address-specific hold
    of the same port, so the bind probe sees it.
    """
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        holder.bind(("127.0.0.1", 0))
        holder.listen(1)
        port = holder.getsockname()[1]

        assert is_port_available(port, "127.0.0.2") is False
    finally:
        holder.close()


def test_is_port_available_still_reports_a_listening_port_taken():
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        holder.bind(("127.0.0.1", 0))
        holder.listen(1)
        port = holder.getsockname()[1]

        assert is_port_available(port, "127.0.0.1") is False
    finally:
        holder.close()


def test_is_port_available_reports_a_free_port_free():
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    assert is_port_available(port, "127.0.0.1") is True


def test_is_port_available_returns_a_bool_for_an_unresolvable_host():
    """Several callers pass an advertised address that may not be local.
    The contract is a bool, never an exception."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    assert isinstance(is_port_available(port, "no-such-host.invalid"), bool)
