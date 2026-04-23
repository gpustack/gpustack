from pathlib import Path

from gpustack.utils import ephemeral_ports


def test_parse_ranges_mixed():
    result = ephemeral_ports._parse_ranges("80, 8000-8010,\t9000")
    assert result == [(80, 80), (8000, 8010), (9000, 9000)]


def test_parse_ranges_empty():
    assert ephemeral_ports._parse_ranges("") == []


def test_merge_adjacent_and_overlap():
    merged = ephemeral_ports._merge(
        [(40000, 40063), (41000, 41999), (41500, 42000), (42001, 42010)]
    )
    assert merged == [(40000, 40063), (41000, 42010)]


def test_format_ranges_single_and_range():
    assert ephemeral_ports._format_ranges([(80, 80), (8000, 8010)]) == "80,8000-8010"


def test_covered_by_true():
    assert ephemeral_ports._covered_by((41005, 41500), [(41000, 41999)])


def test_covered_by_false_partial():
    assert not ephemeral_ports._covered_by((41000, 42500), [(41000, 41999)])


def test_ensure_noop_when_no_overlap(monkeypatch, tmp_path: Path):
    ephemeral = tmp_path / "ip_local_port_range"
    ephemeral.write_text("32768\t60999\n")
    reserved = tmp_path / "ip_local_reserved_ports"
    reserved.write_text("")
    monkeypatch.setattr(ephemeral_ports, "_LOCAL_PORT_RANGE_PATH", ephemeral)
    monkeypatch.setattr(ephemeral_ports, "_RESERVED_PORTS_PATH", reserved)
    monkeypatch.setattr(ephemeral_ports.platform, "system", lambda: "linux")

    ephemeral_ports.ensure_reserved_against_ephemeral(
        [("ray_port_range", (61000, 61999))]
    )

    assert reserved.read_text() == ""


def test_ensure_writes_reservation_on_conflict(monkeypatch, tmp_path: Path):
    ephemeral = tmp_path / "ip_local_port_range"
    ephemeral.write_text("32768\t60999\n")
    reserved = tmp_path / "ip_local_reserved_ports"
    reserved.write_text("")
    monkeypatch.setattr(ephemeral_ports, "_LOCAL_PORT_RANGE_PATH", ephemeral)
    monkeypatch.setattr(ephemeral_ports, "_RESERVED_PORTS_PATH", reserved)
    monkeypatch.setattr(ephemeral_ports.platform, "system", lambda: "linux")

    ephemeral_ports.ensure_reserved_against_ephemeral(
        [
            ("service_port_range", (40000, 40063)),
            ("ray_port_range", (41000, 41999)),
        ]
    )

    assert reserved.read_text() == "40000-40063,41000-41999"


def test_ensure_merges_with_existing_reservation(monkeypatch, tmp_path: Path):
    ephemeral = tmp_path / "ip_local_port_range"
    ephemeral.write_text("32768 60999")
    reserved = tmp_path / "ip_local_reserved_ports"
    reserved.write_text("12345,40000-40100")
    monkeypatch.setattr(ephemeral_ports, "_LOCAL_PORT_RANGE_PATH", ephemeral)
    monkeypatch.setattr(ephemeral_ports, "_RESERVED_PORTS_PATH", reserved)
    monkeypatch.setattr(ephemeral_ports.platform, "system", lambda: "linux")

    ephemeral_ports.ensure_reserved_against_ephemeral(
        [("ray_port_range", (41000, 41999))]
    )

    assert reserved.read_text() == "12345,40000-40100,41000-41999"


def test_ensure_noop_when_already_covered(monkeypatch, tmp_path: Path):
    ephemeral = tmp_path / "ip_local_port_range"
    ephemeral.write_text("32768 60999")
    reserved = tmp_path / "ip_local_reserved_ports"
    reserved.write_text("40000-42000")
    monkeypatch.setattr(ephemeral_ports, "_LOCAL_PORT_RANGE_PATH", ephemeral)
    monkeypatch.setattr(ephemeral_ports, "_RESERVED_PORTS_PATH", reserved)
    monkeypatch.setattr(ephemeral_ports.platform, "system", lambda: "linux")

    ephemeral_ports.ensure_reserved_against_ephemeral(
        [
            ("service_port_range", (40000, 40063)),
            ("ray_port_range", (41000, 41999)),
        ]
    )

    assert reserved.read_text() == "40000-42000"


def test_ensure_aborts_on_unparseable_reserved(monkeypatch, tmp_path: Path, caplog):
    ephemeral = tmp_path / "ip_local_port_range"
    ephemeral.write_text("32768 60999")
    reserved = tmp_path / "ip_local_reserved_ports"
    reserved.write_text("not-a-port-list")
    monkeypatch.setattr(ephemeral_ports, "_LOCAL_PORT_RANGE_PATH", ephemeral)
    monkeypatch.setattr(ephemeral_ports, "_RESERVED_PORTS_PATH", reserved)
    monkeypatch.setattr(ephemeral_ports.platform, "system", lambda: "linux")

    with caplog.at_level("WARNING"):
        ephemeral_ports.ensure_reserved_against_ephemeral(
            [("ray_port_range", (41000, 41999))]
        )

    assert reserved.read_text() == "not-a-port-list"
    assert any("Cannot parse" in rec.message for rec in caplog.records)


def test_parse_ranges_whitespace_separated():
    assert ephemeral_ports._parse_ranges("40000-40063 41000-41999") == [
        (40000, 40063),
        (41000, 41999),
    ]


def test_ensure_warns_when_write_fails(monkeypatch, tmp_path: Path, caplog):
    ephemeral = tmp_path / "ip_local_port_range"
    ephemeral.write_text("32768 60999")
    reserved = tmp_path / "ip_local_reserved_ports"
    reserved.write_text("")
    monkeypatch.setattr(ephemeral_ports, "_LOCAL_PORT_RANGE_PATH", ephemeral)
    monkeypatch.setattr(ephemeral_ports, "_RESERVED_PORTS_PATH", reserved)
    monkeypatch.setattr(ephemeral_ports.platform, "system", lambda: "linux")

    original_write_text = Path.write_text

    def _fail_write(self, *args, **kwargs):
        if self == reserved:
            raise PermissionError("read-only")
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", _fail_write)

    with caplog.at_level("WARNING"):
        ephemeral_ports.ensure_reserved_against_ephemeral(
            [("ray_port_range", (41000, 41999))]
        )

    assert any(
        "overlap the kernel ephemeral port" in rec.message for rec in caplog.records
    )


def test_ensure_skips_when_env_opt_out(monkeypatch, tmp_path: Path):
    ephemeral = tmp_path / "ip_local_port_range"
    ephemeral.write_text("32768 60999")
    reserved = tmp_path / "ip_local_reserved_ports"
    reserved.write_text("")
    monkeypatch.setattr(ephemeral_ports, "_LOCAL_PORT_RANGE_PATH", ephemeral)
    monkeypatch.setattr(ephemeral_ports, "_RESERVED_PORTS_PATH", reserved)
    monkeypatch.setattr(ephemeral_ports.platform, "system", lambda: "linux")
    monkeypatch.setattr(ephemeral_ports.envs, "SKIP_RESERVE_EPHEMERAL_PORTS", True)

    ephemeral_ports.ensure_reserved_against_ephemeral(
        [("ray_port_range", (41000, 41999))]
    )

    assert reserved.read_text() == ""


def test_ensure_skips_non_linux(monkeypatch, tmp_path: Path):
    reserved = tmp_path / "ip_local_reserved_ports"
    reserved.write_text("")
    monkeypatch.setattr(ephemeral_ports, "_RESERVED_PORTS_PATH", reserved)
    monkeypatch.setattr(ephemeral_ports.platform, "system", lambda: "darwin")

    ephemeral_ports.ensure_reserved_against_ephemeral(
        [("ray_port_range", (41000, 41999))]
    )

    assert reserved.read_text() == ""
