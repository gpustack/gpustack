from gpustack.config import registration


class _FakeClientSet:
    def __init__(self, *, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key
        self.http_client = object()


def _registration_client(
    monkeypatch,
    tmp_path,
    token,
    *,
    legacy_uuid=None,
    system_uuid=None,
    existing_worker=False,
):
    clients = []
    if existing_worker:
        (tmp_path / "worker_name").write_text("existing-worker")

    def make_client(**kwargs):
        client = _FakeClientSet(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(registration, "ClientSet", make_client)
    monkeypatch.setattr(registration, "get_legacy_uuid", lambda _: legacy_uuid)
    monkeypatch.setattr(registration, "get_system_uuid", lambda: system_uuid)

    registration.registration_client(
        data_dir=str(tmp_path),
        server_url="http://gpustack.example",
        registration_token=token,
    )
    assert len(clients) == 1
    return clients[0]


def test_fresh_worker_sends_raw_registration_token(monkeypatch, tmp_path):
    client = _registration_client(
        monkeypatch,
        tmp_path,
        "chart-generated-token",
        system_uuid="system-uuid",
    )

    assert client.api_key == "chart-generated-token"


def test_existing_worker_uses_legacy_registration_token(monkeypatch, tmp_path):
    client = _registration_client(
        monkeypatch,
        tmp_path,
        "cluster-token",
        legacy_uuid="worker-uuid",
    )

    assert client.api_key == "gpustack_worker-uuid_cluster-token"


def test_existing_worker_falls_back_to_system_uuid(monkeypatch, tmp_path):
    client = _registration_client(
        monkeypatch,
        tmp_path,
        "cluster-token",
        system_uuid="system-uuid",
        existing_worker=True,
    )

    assert client.api_key == "gpustack_system-uuid_cluster-token"


def test_fresh_worker_falls_back_when_system_uuid_unavailable(monkeypatch, tmp_path):
    def unavailable_system_uuid():
        raise RuntimeError("system UUID unavailable")

    monkeypatch.setattr(registration, "get_legacy_uuid", lambda _: None)
    monkeypatch.setattr(registration, "get_system_uuid", unavailable_system_uuid)
    (tmp_path / "worker_name").write_text("existing-worker")
    clients = []

    def make_client(**kwargs):
        client = _FakeClientSet(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(registration, "ClientSet", make_client)
    registration.registration_client(
        data_dir=str(tmp_path),
        server_url="http://gpustack.example",
        registration_token="chart-generated-token",
    )

    assert clients[0].api_key == "chart-generated-token"


def test_standard_api_key_is_unchanged(monkeypatch, tmp_path):
    token = "gpustack_access-secret"
    client = _registration_client(monkeypatch, tmp_path, token)

    assert client.api_key == token
