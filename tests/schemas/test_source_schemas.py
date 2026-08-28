"""Tests for the three source schema modules — their normalize + reconcile logic:
catalog (``catalog_source.py``), community backends (``inference_backend_source.py``)
and runner overrides (``runner_source.py``)."""

import json
import logging
from importlib.resources import files

import pytest
import yaml
from gpustack_runner.runner import Runner
from pydantic import ValidationError
from sqlalchemy.dialects import mysql, postgresql
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas import runner_source
from gpustack.schemas.catalog_source import (
    CatalogModelEntry,
    CatalogSource,
    _load_catalog,
    build_catalog_entries,
    normalize_catalog_yaml,
    reconcile_catalog,
)
from gpustack.schemas.inference_backend import (
    InferenceBackend,
    VersionConfig,
    VersionConfigDict,
)
from gpustack.schemas.inference_backend_source import (
    InferenceBackendSource,
    compute_disappearing_backend_versions,
    normalize_backend_yaml,
    reconcile_backend,
)
from gpustack.schemas.models import (
    BackendEnum,
    BackendSourceEnum,
    Model,
    SourceEnum,
    SpeculativeConfig,
)
from gpustack.schemas.runner_source import (
    InferenceRunnerSource,
    RunnerOverrideEntry,
    merged_runners,
    merged_service_runners,
    normalize_runner_json,
    proposed_runner_versions,
    reconcile_runner_overrides,
)
from gpustack.schemas.source import SourceContent, SourceTypeEnum, validate_icon

# --- the source model (source.py) -------------------------------------------


@pytest.mark.parametrize(
    "icon",
    [
        None,
        "",
        "https://example.com/logo.png",
        "http://cdn.example.com/logo.png",
        "/static/catalog_icons/qwen.png",  # packaged server-absolute path
        "data:image/png;base64,QQ==",
        "data:image/webp;base64,QQ==",  # any raster data: URI
    ],
)
def test_validate_icon_accepts_urls_paths_and_raster_data(icon):
    assert validate_icon(icon) == icon


@pytest.mark.parametrize(
    "icon",
    [
        "data:image/svg+xml;base64,PHN2Zz48c2NyaXB0Pjwvc2NyaXB0Pjwvc3ZnPg==",
        "data:text/html;base64,PHNjcmlwdD48L3NjcmlwdD4=",  # non-raster data: URI
        "javascript:alert(1)",
        "JavaScript:alert(1)",  # scheme match is case-insensitive
        "java\tscript:alert(1)",  # control chars a browser drops before the scheme
        "vbscript:msgbox",
        "icons/qwen.png",  # relative: nothing resolves it, renders broken
        "//evil.example/x.png",  # protocol-relative: another origin, not a path
    ],
)
def test_validate_icon_rejects_active_content_and_relative_paths(icon):
    with pytest.raises(ValueError):
        validate_icon(icon)


def test_a_source_stores_a_document_larger_than_mysql_text():
    """``TEXT`` caps at 64 KiB on MySQL and every published document is past it,
    so the refresh dies on "Data too long for column 'content'". PostgreSQL and
    SQLite have no limit and keep the type they had.
    """
    for source_cls in (CatalogSource, InferenceBackendSource, InferenceRunnerSource):
        content = source_cls.__table__.c.content.type
        assert content.compile(dialect=mysql.dialect()) == "LONGTEXT"
        assert content.compile(dialect=postgresql.dialect()) == "TEXT"


# --- catalog (catalog_source.py) -------------------------------------------


def _catalog_source(name, source_type, catalog: dict) -> SourceContent:
    """A SourceContent carrying a normalized catalog document."""
    return SourceContent(
        name, source_type, normalize_catalog_yaml(yaml.safe_dump(catalog))
    )


def _spec(repo_id, mode="standard", quantization=None):
    return {
        "source": "huggingface",
        "huggingface_repo_id": repo_id,
        "mode": mode,
        "quantization": quantization,
    }


def _model_set(name, specs, **meta):
    return {"name": name, "specs": specs, **meta}


def _draft(name, repo_id, algorithm="eagle"):
    return {
        "name": name,
        "algorithm": algorithm,
        "source": "huggingface",
        "huggingface_repo_id": repo_id,
    }


def test_normalize_validates_and_defaults_missing_sections():
    # A document with only model_sets normalizes (draft_models defaults to []).
    text = normalize_catalog_yaml(
        yaml.safe_dump({"model_sets": [_model_set("Qwen3", [_spec("Qwen/Qwen3-8B")])]})
    )
    parsed = yaml.safe_load(text)
    assert parsed["draft_models"] == []
    assert parsed["model_sets"][0]["name"] == "Qwen3"

    # Malformed YAML is rejected before persistence, and so is a broken shape:
    # reading ``model_sets: not-a-list`` as "no model sets" would clear the
    # materialized table over a document nobody meant to empty.
    with pytest.raises(ValueError):
        normalize_catalog_yaml("model_sets: [::::")
    with pytest.raises(ValueError):
        normalize_catalog_yaml("model_sets: not-a-list")
    # Only an explicit null means "no model sets"; a mapping read as empty would
    # clear the table just as quietly.
    with pytest.raises(ValueError):
        normalize_catalog_yaml("model_sets: {}")

    # An unreadable record costs itself, not the document. Here the card's only
    # spec fails validation (huggingface source without a repo_id), so the card
    # goes and its neighbour still normalizes.
    text = normalize_catalog_yaml(
        yaml.safe_dump(
            {
                "model_sets": [
                    _model_set("Bad", [{"source": "huggingface"}]),
                    # Broken shape: one dropped card, not a spec per character.
                    {"name": "Shape", "specs": "not-a-list"},
                    _model_set("Good", [_spec("Qwen/Qwen3-8B")]),
                ]
            }
        )
    )
    assert [entry["name"] for entry in yaml.safe_load(text)["model_sets"]] == ["Good"]


def test_normalize_catalog_yaml_is_idempotent():
    """Re-normalizing stored content must be a no-op: the route compares
    normalized text to decide ``changed``. ``gpu_filters`` is the field that
    regressed (a missing-field default serialized as ``null`` until the next pass).
    """
    text = normalize_catalog_yaml(
        yaml.safe_dump(
            {
                "model_sets": [
                    _model_set(
                        "Qwen3",
                        # Only ``vendor`` given, as a bare string: the omitted
                        # ``vendor_variant`` is what used to differ between passes.
                        [
                            dict(
                                _spec("Qwen/Qwen3-8B"),
                                gpu_filters={"vendor": "nvidia"},
                            )
                        ],
                    )
                ]
            }
        )
    )
    assert normalize_catalog_yaml(text) == text
    # ``compute_capability`` is absent, not null: an unset field is dropped.
    assert yaml.safe_load(text)["model_sets"][0]["specs"][0]["gpu_filters"] == {
        "vendor": ["nvidia"],
        "vendor_variant": [],
    }


def test_normalize_catalog_yaml_wires_in_icon_validation():
    """normalize runs model_set icons through validate_icon (full matrix in
    test_sources); here just that it is wired in and passes a valid one."""
    catalog = {
        "model_sets": [
            _model_set("Qwen3", [_spec("Qwen/Qwen3-8B")], icon="/static/qwen.png")
        ]
    }
    parsed = yaml.safe_load(normalize_catalog_yaml(yaml.safe_dump(catalog)))
    assert parsed["model_sets"][0]["icon"] == "/static/qwen.png"

    catalog["model_sets"][0]["icon"] = "javascript:alert(1)"
    with pytest.raises(ValueError):
        normalize_catalog_yaml(yaml.safe_dump(catalog))


def test_packaged_catalog_normalizes(caplog):
    """The baseline the leader seeds must satisfy the validation it ships with —
    its icons are all ``/static/catalog_icons/...`` paths."""
    raw = files("gpustack.assets").joinpath("model-catalog.yaml").read_text()
    caplog.set_level(logging.WARNING)
    text = normalize_catalog_yaml(raw)
    # Our own document is not from the future: the leader re-seeds it on every
    # start, and an admin may PUT a hand-edited copy of it back as a FILE source.
    # Its ``.``-prefixed anchor hosts sit inside cards, not only at the top level.
    assert not caplog.text
    normalize_catalog_yaml(raw, strict=True)
    assert normalize_catalog_yaml(text) == text  # idempotent on the real catalog
    parsed = yaml.safe_load(text)
    assert len(parsed["model_sets"]) > 100
    assert all(
        model_set["icon"].startswith("/static/")
        for model_set in parsed["model_sets"]
        if model_set.get("icon")
    )
    # Dropping unset fields must not change what the document means: they parse
    # back to None either way. Asserted on the real catalog because a field whose
    # default stops being None would start losing an explicit null silently.
    assert ": null" not in text
    assert _load_catalog(text) == _load_catalog(raw)

    modelscope = files("gpustack.assets").joinpath("model-catalog-modelscope.yaml")
    caplog.clear()
    normalize_catalog_yaml(modelscope.read_text(), strict=True)
    assert not caplog.text


def test_build_entries_merges_specs_and_stamps_last_writer_source():
    src_a = _catalog_source(
        "a",
        SourceTypeEnum.BUILTIN,
        {
            "model_sets": [_model_set("Qwen3", [_spec("Qwen/Qwen3-8B")], order=1)],
            "draft_models": [_draft("eagle-qwen", "draft/v1")],
        },
    )
    src_b = _catalog_source(
        "b",
        SourceTypeEnum.FILE,
        {
            "model_sets": [
                # Same set name: new spec unions in, duplicate identity deduped,
                # metadata + source from the last writer.
                _model_set(
                    "Qwen3",
                    [_spec("Qwen/Qwen3-8B"), _spec("Qwen/Qwen3-32B")],
                    order=5,
                )
            ],
            "draft_models": [_draft("eagle-qwen", "draft/v2")],
        },
    )

    entries = {(e.kind, e.name): e for e in build_catalog_entries([src_a, src_b])}

    model_set = entries[("model_set", "Qwen3")]
    repo_ids = {spec["huggingface_repo_id"] for spec in model_set.payload["specs"]}
    assert repo_ids == {"Qwen/Qwen3-8B", "Qwen/Qwen3-32B"}  # union, deduped
    assert model_set.payload["order"] == 5  # last writer's metadata
    assert (model_set.source_name, model_set.source_type) == ("b", SourceTypeEnum.FILE)

    draft = entries[("draft", "eagle-qwen")]
    assert draft.payload["huggingface_repo_id"] == "draft/v2"
    assert draft.source_name == "b"


@pytest.mark.asyncio
async def test_reconcile_keeps_ids_stable_and_rewrites():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(
            SQLModel.metadata.create_all,
            tables=[CatalogModelEntry.__table__, Model.__table__],
        )

    async def _entries(session):
        rows = await CatalogModelEntry.all(session)
        return {(r.kind, r.name): r for r in rows}

    async with AsyncSession(engine) as session:
        first = _catalog_source(
            "a",
            SourceTypeEnum.FILE,
            {
                "model_sets": [
                    _model_set("Qwen3", [_spec("Qwen/Qwen3-8B")]),
                    _model_set("Llama", [_spec("meta/Llama-8B")]),
                ],
                "draft_models": [_draft("eagle", "draft/v1")],
            },
        )
        await reconcile_catalog(session, [first])
        before = await _entries(session)
        qwen_id = before[("model_set", "Qwen3")].id
        assert set(before) == {
            ("model_set", "Qwen3"),
            ("model_set", "Llama"),
            ("draft", "eagle"),
        }

        # Second reconcile: Qwen3 stays (id must not churn), Llama drops, Gemma added.
        second = _catalog_source(
            "a",
            SourceTypeEnum.FILE,
            {
                "model_sets": [
                    _model_set("Qwen3", [_spec("Qwen/Qwen3-8B")], order=9),
                    _model_set("Gemma", [_spec("google/gemma-2b")]),
                ],
                "draft_models": [],
            },
        )
        await reconcile_catalog(session, [second])
        after = await _entries(session)

        assert set(after) == {("model_set", "Qwen3"), ("model_set", "Gemma")}
        assert after[("model_set", "Qwen3")].id == qwen_id  # stable across reconcile
        assert after[("model_set", "Qwen3")].payload["order"] == 9

    await engine.dispose()


@pytest.mark.asyncio
async def test_a_dropped_draft_model_survives_only_while_a_deployment_pins_it(caplog):
    """Written as the likely scenario rather than an upstream mistake:
    ``draft_models`` is optional and normalizes to ``[]``, so an admin writing a
    catalog of their own takes the packaged ones away without ever seeing the key.
    Model sets get no such treatment."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(
            SQLModel.metadata.create_all,
            tables=[CatalogModelEntry.__table__, Model.__table__],
        )

    async with AsyncSession(engine) as session:

        async def keys():
            return {
                (row.kind, row.name) for row in await CatalogModelEntry.all(session)
            }

        await reconcile_catalog(
            session,
            [
                _catalog_source(
                    "builtin",
                    SourceTypeEnum.BUILTIN,
                    {
                        "model_sets": [_model_set("Qwen3", [_spec("Qwen/Qwen3-8B")])],
                        "draft_models": [
                            _draft("eagle-deployed", "draft/deployed"),
                            _draft("eagle-idle", "draft/idle"),
                        ],
                    },
                )
            ],
        )
        await Model.create(
            session,
            Model(
                name="pinned-model",
                source=SourceEnum.HUGGING_FACE,
                huggingface_repo_id="org/repo",
                speculative_config=SpeculativeConfig(
                    enabled=True, draft_model="eagle-deployed"
                ),
                replicas=1,
            ),
        )

        # A custom catalog carrying only model sets: draft_models is absent, which
        # is indistinguishable from an empty one by the time it is stored.
        custom = _catalog_source(
            "custom",
            SourceTypeEnum.URL,
            {"model_sets": [_model_set("Gemma", [_spec("google/gemma-2b")])]},
        )
        with caplog.at_level(logging.WARNING):
            await reconcile_catalog(session, [custom])
        assert await keys() == {
            ("model_set", "Gemma"),
            ("draft", "eagle-deployed"),
        }
        assert "pinned-model" in caplog.text

        # Once nothing pins it, the next reconcile lets it go.
        model = await Model.one_by_field(session, "name", "pinned-model")
        await model.delete(session)
        await reconcile_catalog(session, [custom])
        assert await keys() == {("model_set", "Gemma")}

    await engine.dispose()


# --- community backends (inference_backend_source.py) ----------------------


def _backend(name, versions, **extra):
    """A community backend config dict as it appears in the YAML source."""
    return {
        "backend_name": name,
        "version_configs": {
            version: {"image_name": image} for version, image in versions.items()
        },
        **extra,
    }


def _backend_source(name, *configs) -> SourceContent:
    """A SourceContent carrying a normalized community-backend document."""
    return SourceContent(
        name, SourceTypeEnum.FILE, normalize_backend_yaml(yaml.safe_dump(list(configs)))
    )


def test_normalize_backend_yaml_validates():
    # A valid document round-trips through normalize (list of backend configs).
    text = normalize_backend_yaml(yaml.safe_dump([_backend("foo", {"v1": "img:v1"})]))
    parsed = yaml.safe_load(text)
    assert parsed[0]["backend_name"] == "foo"
    assert parsed[0]["version_configs"]["v1"]["image_name"] == "img:v1"

    # Malformed input is rejected before persistence.
    with pytest.raises(ValueError):
        normalize_backend_yaml("backend_name: [::::")  # not valid YAML
    with pytest.raises(ValueError):
        normalize_backend_yaml(yaml.safe_dump({"backend_name": "foo"}))  # not a list
    with pytest.raises(ValueError):
        normalize_backend_yaml(
            yaml.safe_dump([{"version_configs": {"v1": {}}}])
        )  # no name
    with pytest.raises(ValueError):
        normalize_backend_yaml(yaml.safe_dump([_backend("foo", {})]))  # empty versions
    with pytest.raises(ValueError):
        # version without an image_name
        normalize_backend_yaml(
            yaml.safe_dump([{"backend_name": "foo", "version_configs": {"v1": {}}}])
        )
    # A built-in engine name is reserved: the upsert resolves an existing row by
    # backend_name alone, so this document would take over that engine's row.
    for reserved in ("vLLM", "sglang"):
        with pytest.raises(ValueError, match="built-in engine"):
            normalize_backend_yaml(
                yaml.safe_dump([_backend(reserved, {"v1": "img:v1"})])
            )


def test_normalize_backend_yaml_wires_in_icon_validation():
    """normalize runs backend icons through validate_icon (full matrix in
    test_sources); here just that it is wired in and passes a valid one."""
    passed = normalize_backend_yaml(
        yaml.safe_dump([_backend("foo", {"v1": "img:v1"}, icon="/static/foo.png")])
    )
    assert yaml.safe_load(passed)[0]["icon"] == "/static/foo.png"

    with pytest.raises(ValueError):
        normalize_backend_yaml(
            yaml.safe_dump(
                [_backend("foo", {"v1": "img:v1"}, icon="javascript:alert(1)")]
            )
        )


def test_packaged_community_backends_normalize(caplog):
    """The baseline the leader seeds must satisfy the validation it ships with —
    its icons are all inline ``data:image/png`` URIs."""
    raw = files("gpustack.assets").joinpath("community-inference-backends.yaml")
    caplog.set_level(logging.WARNING)
    configs = yaml.safe_load(normalize_backend_yaml(raw.read_text()))
    # Our own document is not from the future: it is dumped from the read API, so
    # it carries fields (``is_built_in``, ``framework_index_map``) the write side
    # drops on purpose. Recognized ≠ materialized.
    assert not caplog.text
    normalize_backend_yaml(raw.read_text(), strict=True)
    assert any("is_built_in" in config for config in configs)
    icons = [config["icon"] for config in configs if config.get("icon")]
    # `make install` rebuilds this file from the community-inference-backends
    # repo, so how many entries it carries is not ours to pin.
    assert icons
    assert all(icon.startswith("data:image/png;base64,") for icon in icons)


def test_compute_disappearing_backend_versions():
    """The proposed source set is re-merged with the same union
    reconcile_backend uses; whatever drops out of the union but is still in
    the caller's current-versions snapshot is reported as disappearing."""
    builtin = _backend_source("builtin", _backend("foo", {"v1": "img:v1"}))
    custom = _backend_source("custom", _backend("bar", {"v1": "bar:v1"}))

    # Deleting a custom source: its versions disappear, the baseline's don't.
    assert compute_disappearing_backend_versions(
        [builtin], {("foo", "v1"), ("bar", "v1")}
    ) == {("bar", "v1")}

    # Turning off the builtin baseline: the baseline's versions disappear.
    assert compute_disappearing_backend_versions(
        [custom], {("foo", "v1"), ("bar", "v1")}
    ) == {("foo", "v1")}

    # PUT with new content that drops an older version.
    replaced = _backend_source("custom", _backend("foo", {"v2": "img:v2"}))
    assert compute_disappearing_backend_versions(
        [replaced], {("foo", "v1"), ("foo", "v2")}
    ) == {("foo", "v1")}


@pytest.mark.asyncio
async def test_reconcile_backend_smart_merge_and_cleanup():
    """reconcile_backend materializes the union of all sources into the
    Platform-NULL community rows: it preserves user modifications, unions
    versions across sources, cleans up backends no longer present, and never
    touches Org-private rows or built-in engine rows."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(
            SQLModel.metadata.create_all, tables=[InferenceBackend.__table__]
        )

    async with AsyncSession(engine) as session:
        # Rows the source pipeline must never touch.
        await InferenceBackend.create(
            session,
            InferenceBackend(
                backend_name="vllm",
                is_built_in=True,
                backend_source=BackendSourceEnum.BUILT_IN,
                enabled=True,
            ),
        )
        await InferenceBackend.create(
            session,
            InferenceBackend(
                backend_name="foo",
                owner_principal_id=1,  # Org-private extension of the same name
                backend_source=BackendSourceEnum.COMMUNITY,
                enabled=True,
            ),
        )

        # First reconcile from a single source: the four appear as community.
        await reconcile_backend(
            session,
            [
                _backend_source(
                    "a",
                    # ``is_built_in`` is a field we recognize but never write: a
                    # community card claiming it would shortcut version resolution
                    # and read as an engine GPUStack ships.
                    _backend("foo", {"v1": "img:v1"}, is_built_in=True),
                    _backend("bar", {"v1": "bar:v1"}),
                    _backend("baz", {"v1": "baz:v1"}),
                    _backend("qux", {"v1": "qux:v1"}),
                )
            ],
        )

        async def _platform(name):
            return await InferenceBackend.one_by_fields(
                session, {"backend_name": name, "owner_principal_id": None}
            )

        foo = await _platform("foo")
        assert foo.backend_source == BackendSourceEnum.COMMUNITY
        assert foo.enabled is False
        assert foo.is_built_in is False
        assert foo.source_name == "a" and foo.source_type == SourceTypeEnum.FILE

        # The user enables foo, adds a custom version, sets default_env; enables bar.
        foo_versions = dict(foo.version_configs.root)
        foo_versions["x-custom"] = VersionConfig(
            image_name="img:x", built_in_frameworks=None
        )
        await foo.update(
            session,
            {
                "enabled": True,
                "default_env": {"K": "V"},
                "version_configs": VersionConfigDict(root=foo_versions),
            },
        )
        bar = await _platform("bar")
        await bar.update(session, {"enabled": True})
        # baz stays disabled but gains a hand-added version; qux is untouched.
        baz = await _platform("baz")
        baz_versions = dict(baz.version_configs.root)
        baz_versions["y-custom"] = VersionConfig(
            image_name="baz:y", built_in_frameworks=None
        )
        await baz.update(
            session, {"version_configs": VersionConfigDict(root=baz_versions)}
        )

        # Second reconcile: foo now has v2 from a second source; the rest
        # disappear.
        await reconcile_backend(
            session,
            [
                _backend_source("a", _backend("foo", {"v1": "img:v1"})),
                _backend_source("b", _backend("foo", {"v2": "img:v2"})),
            ],
        )

        foo = await _platform("foo")
        # User modifications survive the reconcile.
        assert foo.enabled is True
        assert foo.default_env == {"K": "V"}
        # Versions: union across sources + the user's custom version.
        assert set(foo.version_configs.root.keys()) == {"v1", "v2", "x-custom"}
        # Card source = the last writer that produced this backend.
        assert foo.source_name == "b"
        # Per-version source: v1 came from "a", v2 from "b", and the user's
        # custom version carries no source.
        assert foo.version_configs.root["v1"].source_name == "a"
        assert foo.version_configs.root["v2"].source_name == "b"
        assert foo.version_configs.root["v2"].source_type == SourceTypeEnum.FILE
        assert foo.version_configs.root["x-custom"].source_name is None

        # bar was enabled → converted to a custom backend, not deleted.
        bar = await _platform("bar")
        assert bar is not None
        assert bar.backend_source == BackendSourceEnum.CUSTOM
        assert bar.enabled is False

        # baz was disabled but carried a hand-added version, which no in-use
        # check covers → also converted rather than deleted, versions intact.
        baz = await _platform("baz")
        assert baz is not None
        assert baz.backend_source == BackendSourceEnum.CUSTOM
        assert set(baz.version_configs.root.keys()) == {"v1", "y-custom"}
        # qux carried nothing of the user's → deleted.
        assert await _platform("qux") is None

        # A source publishing baz again adopts the converted row back, keeping
        # the hand-added version alongside the source's own.
        await reconcile_backend(
            session,
            [
                _backend_source(
                    "a",
                    _backend("foo", {"v1": "img:v1"}),
                    _backend("baz", {"v1": "baz:v1"}),
                )
            ],
        )
        baz = await _platform("baz")
        assert baz.backend_source == BackendSourceEnum.COMMUNITY
        assert baz.source_name == "a"
        assert set(baz.version_configs.root.keys()) == {"v1", "y-custom"}
        assert baz.version_configs.root["v1"].built_in_frameworks == []
        assert baz.version_configs.root["y-custom"].built_in_frameworks is None

        # Untouched: built-in engine row and the Org-private row.
        vllm = await _platform("vllm")
        assert vllm.backend_source == BackendSourceEnum.BUILT_IN
        org_foo = await InferenceBackend.one_by_fields(
            session, {"backend_name": "foo", "owner_principal_id": 1}
        )
        assert org_foo is not None and org_foo.owner_principal_id == 1

    await engine.dispose()


@pytest.mark.asyncio
async def test_a_backend_card_this_version_cannot_model_costs_only_itself(caplog):
    """A card the table model takes but cannot read back costs the whole round.
    Refused before the session is touched, its neighbours land.

    Both failure modes: pydantic's ``ValidationError``, and the ``TypeError``
    ``**`` raises on a non-string key — catching only the first let it escape.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(
            SQLModel.metadata.create_all, tables=[InferenceBackend.__table__]
        )

    async with AsyncSession(engine) as session:

        async def _platform(name):
            return await InferenceBackend.one_by_fields(
                session, {"backend_name": name, "owner_principal_id": None}
            )

        await reconcile_backend(
            session,
            [
                _backend_source(
                    "a",
                    _backend("aaa", {"v1": "img:v1"}),
                    _backend("zzz", {"v1": "old:v1"}),
                    _backend("mmm", {"v1": "old:v1"}),
                )
            ],
        )

        # zzz gains an unknown parameter_format, mmm a non-string version key.
        non_string_key = _backend("mmm", {"v1": "new:v1"})
        non_string_key["version_configs"]["v1"][7] = "x"
        with caplog.at_level(logging.ERROR):
            await reconcile_backend(
                session,
                [
                    _backend_source(
                        "a",
                        _backend("aaa", {"v2": "img:v2"}),
                        _backend(
                            "zzz", {"v1": "new:v1"}, parameter_format="future-format"
                        ),
                        non_string_key,
                    )
                ],
            )

        # The healthy card's update landed: the round used to be lost entirely.
        aaa = await _platform("aaa")
        assert set(aaa.version_configs.root.keys()) == {"v2"}
        # Kept, not deleted: both stay in the merged union.
        zzz = await _platform("zzz")
        assert zzz.version_configs.root["v1"].image_name == "old:v1"
        assert zzz.parameter_format is None
        assert zzz.backend_source == BackendSourceEnum.COMMUNITY
        mmm = await _platform("mmm")
        assert mmm.version_configs.root["v1"].image_name == "old:v1"
        assert "zzz" in caplog.text and "mmm" in caplog.text

    await engine.dispose()


@pytest.mark.asyncio
async def test_a_write_failure_is_not_mistaken_for_a_skippable_card(monkeypatch):
    """The per-card ``except`` must not swallow a failure from the write itself:
    that one has already rolled the round back, so continuing past it commits a
    part-way materialization. Easy to get wrong because pydantic's
    ``ValidationError`` is a ``ValueError``, which the write path can raise too.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(
            SQLModel.metadata.create_all, tables=[InferenceBackend.__table__]
        )

    async def failing_create(*args, **kwargs):
        VersionConfig(built_in_frameworks="not-a-list")

    monkeypatch.setattr(
        InferenceBackend, "create", classmethod(lambda cls, *a, **k: failing_create())
    )
    async with AsyncSession(engine) as session:
        with pytest.raises(ValidationError):
            await reconcile_backend(
                session, [_backend_source("a", _backend("foo", {"v1": "img:v1"}))]
            )

    await engine.dispose()


# --- runner overrides (runner_source.py) -----------------------------------

# A service the packaged gpustack-runner catalog never carries, so a reconcile
# assertion never depends on whatever versions the installed package ships.
_SYNTHETIC_SERVICE = "synthetic-service"


def _entry(service_version):
    return {
        "backend": "cuda",
        "service": _SYNTHETIC_SERVICE,
        "service_version": service_version,
        "platform": "linux/amd64",
        "docker_image": f"img:{service_version}",
    }


def _runner_source(name, source_type, *entries) -> SourceContent:
    return SourceContent(name, source_type, json.dumps(list(entries)))


def _runner(service_version, docker_image) -> Runner:
    """A packaged runner differing from its peers only by version and image."""
    return Runner(
        backend="cuda",
        backend_version="12.4",
        original_backend_version="12.4",
        backend_variant="",
        service="vllm",
        service_version=service_version,
        platform="linux/amd64",
        docker_image=docker_image,
        deprecated=False,
    )


def _override(
    service_version, docker_image, source_type=SourceTypeEnum.OFFICIAL
) -> RunnerOverrideEntry:
    """An override row on the same natural key ``_runner`` produces."""
    entry = runner_source._entry_from_runner(_runner(service_version, docker_image))
    entry.source_type = source_type
    return entry


def test_override_rows_replace_the_packaged_runner_catalog(monkeypatch):
    """Whole-content replacement, as ``order_source_contents`` does it for the other
    two kinds: any override row means a remote document is in service, so the
    packaged catalog steps aside — the official slot as much as a custom source,
    which is what lets a coordinate be *withdrawn*.

    ``list_runners`` is stubbed so the assertions don't depend on the installed
    package's versions.
    """
    packaged = [_runner("0.11.0", "pkg:0.11.0"), _runner("0.10.0", "pkg:0.10.0")]
    monkeypatch.setattr(runner_source, "list_runners", lambda **filters: packaged)

    def images(overrides):
        return {
            runner.service_version: runner.docker_image
            for runner in merged_runners(overrides)
        }

    # No override row: the packaged catalog is the answer.
    assert images([]) == {"0.11.0": "pkg:0.11.0", "0.10.0": "pkg:0.10.0"}

    # The official document corrects 0.11.0, adds 0.12.0 — and withdraws 0.10.0
    # simply by not carrying it.
    official = [
        _override("0.11.0", "official:0.11.0-fixed"),
        _override("0.12.0", "official:0.12.0"),
    ]
    assert images(official) == {
        "0.11.0": "official:0.11.0-fixed",
        "0.12.0": "official:0.12.0",
    }

    # A custom document behaves identically: the rows are already the merge of
    # every enabled source, so nothing downstream distinguishes the two.
    custom = _override("0.12.0", "custom:0.12.0", SourceTypeEnum.URL)
    assert images([custom]) == {"0.12.0": "custom:0.12.0"}

    # The decision is whether any remote document is in service, not what survives
    # the filter: a scoped query its document has nothing for comes back empty
    # rather than falling back to the package.
    assert merged_runners([custom], backend="rocm") == []

    # The grouped view the scheduler's backend filter consumes sees the same
    # replacement (that filter's own tests stub this call out).
    grouped = merged_service_runners([custom])
    assert [
        (service.service, [version.version for version in service.versions])
        for service in grouped
    ] == [("vllm", ["0.12.0"])]


def _vllm_entry(service_version, platform="linux/amd64"):
    """An entry on a real built-in backend's ``service``, so the reconcile's name
    bridge resolves it — ``_SYNTHETIC_SERVICE`` is deliberately not a built-in."""
    return {
        "backend": "cuda",
        "service": BackendEnum.VLLM.value.lower(),
        "service_version": service_version,
        "platform": platform,
        "docker_image": f"img:{service_version}",
    }


@pytest.mark.asyncio
async def test_reconcile_rewrites_the_table_from_the_merged_sources():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(
            SQLModel.metadata.create_all,
            tables=[RunnerOverrideEntry.__table__, Model.__table__],
        )

    async with AsyncSession(engine) as session:
        await reconcile_runner_overrides(
            session,
            [
                _runner_source("official", SourceTypeEnum.OFFICIAL, _entry("9.9.9")),
            ],
        )
        rows = {
            row.service_version: row for row in await RunnerOverrideEntry.all(session)
        }
        assert set(rows) == {"9.9.9"}
        assert rows["9.9.9"].docker_image == "img:9.9.9"
        assert rows["9.9.9"].source_name == "official"
        assert rows["9.9.9"].source_type == SourceTypeEnum.OFFICIAL

        # Full rewrite: a dropped version is gone, and every row carries the
        # source that produced it (last writer wins a shared natural key).
        await reconcile_runner_overrides(
            session,
            [
                _runner_source("official", SourceTypeEnum.OFFICIAL, _entry("8.8.8")),
                _runner_source(
                    "custom", SourceTypeEnum.URL, _entry("8.8.8"), _entry("7.7.7")
                ),
            ],
        )
        rows = {
            row.service_version: row for row in await RunnerOverrideEntry.all(session)
        }
        assert set(rows) == {"8.8.8", "7.7.7"}
        assert rows["8.8.8"].source_name == "custom"
        assert rows["8.8.8"].source_type == SourceTypeEnum.URL

    await engine.dispose()


@pytest.mark.asyncio
async def test_a_dropped_coordinate_survives_only_while_a_deployment_pins_it(caplog):
    """Survival takes the whole coordinate (a row per platform), spares nothing
    nobody deploys, does not hold back the rest of the document, and is announced
    rather than silent. It ends the moment the deployment does."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(
            SQLModel.metadata.create_all,
            tables=[RunnerOverrideEntry.__table__, Model.__table__],
        )

    async with AsyncSession(engine) as session:

        async def platforms_by_version():
            grouped = {}
            for row in await RunnerOverrideEntry.all(session):
                grouped.setdefault(row.service_version, set()).add(row.platform)
            return grouped

        await reconcile_runner_overrides(
            session,
            [
                _runner_source(
                    "official",
                    SourceTypeEnum.OFFICIAL,
                    _vllm_entry("0.11.0", "linux/amd64"),
                    _vllm_entry("0.11.0", "linux/arm64"),
                    _vllm_entry("0.10.0"),
                )
            ],
        )
        await Model.create(
            session,
            Model(
                name="pinned-model",
                source=SourceEnum.HUGGING_FACE,
                huggingface_repo_id="org/repo",
                backend=BackendEnum.VLLM.value,
                backend_version="0.11.0",
                replicas=1,
            ),
        )

        # The document drops both old versions and publishes a new one. 0.11.0 is
        # deployed, so both of its platform rows stay; 0.10.0 is nobody's, so it
        # goes; 0.12.0 lands either way — the deployment does not freeze the update.
        with caplog.at_level(logging.WARNING):
            await reconcile_runner_overrides(
                session,
                [
                    _runner_source(
                        "official", SourceTypeEnum.OFFICIAL, _vllm_entry("0.12.0")
                    )
                ],
            )
        assert await platforms_by_version() == {
            "0.11.0": {"linux/amd64", "linux/arm64"},
            "0.12.0": {"linux/amd64"},
        }
        assert "pinned-model" in caplog.text

        # Once nothing pins it, the next reconcile lets it go.
        model = await Model.one_by_field(session, "name", "pinned-model")
        await model.delete(session)
        await reconcile_runner_overrides(
            session,
            [
                _runner_source(
                    "official", SourceTypeEnum.OFFICIAL, _vllm_entry("0.12.0")
                )
            ],
        )
        assert await platforms_by_version() == {"0.12.0": {"linux/amd64"}}

    await engine.dispose()


@pytest.mark.asyncio
async def test_falling_back_to_the_packaged_catalog_keeps_nothing():
    """Keeping a survivor here would defeat the fall-back it looks like it
    protects: ``merged_runners`` steps the packaged catalog aside for any row at
    all, so one row would leave the cluster on that single coordinate instead of
    the complete baseline."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(
            SQLModel.metadata.create_all,
            tables=[RunnerOverrideEntry.__table__, Model.__table__],
        )

    async with AsyncSession(engine) as session:
        await reconcile_runner_overrides(
            session,
            [
                _runner_source(
                    "official",
                    SourceTypeEnum.OFFICIAL,
                    _vllm_entry("0.11.0"),
                    _vllm_entry("0.9.0"),
                )
            ],
        )
        await Model.create(
            session,
            Model(
                name="pinned-model",
                source=SourceEnum.HUGGING_FACE,
                huggingface_repo_id="org/repo",
                backend=BackendEnum.VLLM.value,
                backend_version="0.11.0",
                replicas=1,
            ),
        )

        # What ``remote_enabled: false`` comes to: every source out of service, so
        # the merge input is empty.
        await reconcile_runner_overrides(session, [])
        assert await RunnerOverrideEntry.all(session) == []

    await engine.dispose()


def test_normalize_runner_json_leaves_the_source_stamp_out():
    """The stamp is row metadata, not document content: leaking it into the
    canonical text would change every source's ``content_hash`` and defeat the
    "download the baseline, PUT it back unchanged" round-trip."""
    text = normalize_runner_json(json.dumps([_entry("1.0.0")]))
    assert "source_name" not in text and "source_type" not in text


# --- shared: one unreadable document ---------------------------------------


def test_an_unreadable_source_is_skipped_without_taking_out_the_kind():
    """A document only fails to parse at merge time after a downgrade or a schema
    change, where the kind's other sources still serve. All three isolate per
    source; all of them unreadable raises instead.
    """
    unreadable = SourceContent("stale", SourceTypeEnum.FILE, "{{ not a document")

    catalog = _catalog_source(
        "good", SourceTypeEnum.BUILTIN, {"model_sets": [_model_set("Qwen", [])]}
    )
    entries = build_catalog_entries([unreadable, catalog])
    assert [entry.name for entry in entries] == ["Qwen"]

    runner = _runner_source("good", SourceTypeEnum.OFFICIAL, _entry("1.0.0"))
    assert proposed_runner_versions([unreadable, runner]) >= {
        (_SYNTHETIC_SERVICE, "1.0.0")
    }

    # The community-backend merge, reached through its public caller: the good
    # source's version is in the proposed union, so it is not "disappearing".
    backend = _backend_source("good", _backend("foo", {"v1": "img:v1"}))
    assert (
        compute_disappearing_backend_versions([unreadable, backend], {("foo", "v1")})
        == set()
    )

    # When *nothing* is readable, the merge raises rather than answering "empty":
    # an empty answer is indistinguishable from a document that deliberately
    # carries nothing, and the caller would clear a materialized table over stored
    # text nobody chose.
    with pytest.raises(ValueError, match="no catalog source could be read"):
        build_catalog_entries([unreadable])
    with pytest.raises(ValueError, match="no runner source could be read"):
        proposed_runner_versions([unreadable])
    with pytest.raises(ValueError, match="no backend source could be read"):
        compute_disappearing_backend_versions([unreadable], set())


def test_a_non_string_yaml_key_is_an_unknown_field_not_a_crash():
    """A YAML key need not be a string. Read as the unknown field it looks like,
    at every level: the ``AttributeError``/``TypeError`` it used to raise escapes
    the per-source ``except ValueError`` and takes out the whole round.
    """
    for document in (
        {1: "x", "model_sets": [_model_set("Q", [_spec("x/y")])]},
        {"model_sets": [{**_model_set("Q", [_spec("x/y")]), 1: "x"}]},
        {"model_sets": [_model_set("Q", [{**_spec("x/y"), 1: "x"}])]},
    ):
        catalog = yaml.safe_dump(document)
        assert {
            entry.name
            for entry in build_catalog_entries(
                [SourceContent("newer", SourceTypeEnum.URL, catalog)]
            )
        } == {"Q"}
        with pytest.raises(ValueError, match=r"unknown catalog field\(s\): 1"):
            normalize_catalog_yaml(catalog, strict=True)

    backend = yaml.safe_dump([{**_backend("foo", {"v1": "img:v1"}), 7: "x"}])
    assert normalize_backend_yaml(backend)
    with pytest.raises(ValueError, match=r"unknown community backend field\(s\): 7"):
        normalize_backend_yaml(backend, strict=True)


def test_a_document_from_a_newer_gpustack_still_loads(caplog):
    """A field or enum value added after this release must cost its own record,
    never the whole document — that would freeze updates on every older
    installation. All three kinds drop it, name it once, and serve the rest.
    """
    caplog.set_level(logging.WARNING)

    # catalog: an added top-level key, an unreadable spec, a card on an unknown
    # size_unit, and an unreadable draft.
    document = yaml.safe_dump(
        {
            "model_categories": ["something-new"],
            # ``.``-prefixed keys host YAML anchors, not fields to warn about.
            ".pinned_version": "0.1.0",
            "model_sets": [
                _model_set(
                    "Qwen3",
                    [
                        dict(_spec("Qwen/Qwen3-8B"), source="future-hub"),
                        dict(
                            _spec("Qwen/Qwen3-32B"),
                            future_knob="x",
                            **{".spec_anchor": "s"},
                        ),
                    ],
                    future_card_field="y",
                    **{".card_anchor": "c"},
                ),
                _model_set(
                    "Odd", [_spec("x/y")], size_unit="Q", dropped_card_field="z"
                ),
            ],
            "draft_models": [
                dict(_draft("future-draft", "x/d"), source="future-hub"),
                dict(_draft("good-draft", "x/good"), **{".draft_anchor": "d"}),
            ],
        }
    )
    entries = {
        (entry.kind, entry.name): entry
        for entry in build_catalog_entries(
            [SourceContent("newer", SourceTypeEnum.OFFICIAL, document)]
        )
    }
    assert set(entries) == {("model_set", "Qwen3"), ("draft", "good-draft")}
    # A partly readable card keeps its readable specs, not losing a deployable
    # model to an unreadable quantization.
    assert [
        spec["huggingface_repo_id"]
        for spec in entries[("model_set", "Qwen3")].payload["specs"]
    ] == ["Qwen/Qwen3-32B"]
    # Still idempotent: content_hash decides whether a refresh writes at all.
    text = normalize_catalog_yaml(document)
    assert normalize_catalog_yaml(text) == text
    # Where the leniency stops: keeping none is not an empty catalog.
    with pytest.raises(ValueError, match="none of the records"):
        normalize_catalog_yaml(
            yaml.safe_dump(
                {"model_sets": [_model_set("Odd", [{"source": "future-hub"}])]}
            )
        )

    # runner: the kind that used to reject the whole document over one added key.
    runner = _runner_source(
        "newer", SourceTypeEnum.OFFICIAL, dict(_entry("1.0.0"), runtime_flavor="new")
    )
    assert (_SYNTHETIC_SERVICE, "1.0.0") in proposed_runner_versions([runner])
    assert "runtime_flavor" not in normalize_runner_json(runner.content)

    # community backend: an added card-level key.
    backend = _backend_source(
        "newer", _backend("foo", {"v1": "img:v1"}, telemetry_endpoint="x")
    )
    assert compute_disappearing_backend_versions([backend], {("foo", "v1")}) == set()
    # Only this kind keeps it in the stored content, so an upgrade picks it up.
    assert "telemetry_endpoint" in backend.content

    # Each kind names what it ignored once per document, the catalog at every level.
    assert "model_categories" in caplog.text
    assert "future_card_field" in caplog.text
    assert "future_knob" in caplog.text
    assert "runtime_flavor" in caplog.text
    assert "telemetry_endpoint" in caplog.text
    # A ``.``-prefixed key hosts a YAML anchor at whichever mapping carries it —
    # the packaged catalog puts them at the top and inside cards.
    assert not any(
        anchor in caplog.text
        for anchor in ("pinned_version", "card_anchor", "spec_anchor", "draft_anchor")
    )
    # A dropped card stays out of the field report; it already warned.
    assert "dropped_card_field" not in caplog.text


def test_the_same_document_is_refused_on_the_way_in_through_the_api():
    """Two contracts on one document: dropped when read unattended, named when
    PUT through the configuration API, where an admin is waiting on the answer.
    """
    catalog = yaml.safe_dump(
        {
            "model_categories": ["something-new"],
            ".top_anchor": "t",
            "model_sets": [
                _model_set("Qwen3", [_spec("Qwen/Qwen3-8B")], **{".card_anchor": "c"})
            ],
        }
    )
    assert normalize_catalog_yaml(catalog)  # unattended: the field goes, the card stays
    with pytest.raises(ValueError, match=r"field\(s\): model_categories$"):
        normalize_catalog_yaml(catalog, strict=True)

    # Every level: a field inside a card or spec is dropped from stored content too.
    nested = yaml.safe_dump(
        {
            "model_sets": [
                _model_set(
                    "Qwen3",
                    [dict(_spec("Qwen/Qwen3-8B"), future_knob="x")],
                    future_card_field="y",
                )
            ]
        }
    )
    assert normalize_catalog_yaml(nested)
    with pytest.raises(ValueError, match="future_card_field, future_knob"):
        normalize_catalog_yaml(nested, strict=True)

    # A record, not a field: the case an admin hits with a typo.
    typo = yaml.safe_dump(
        {
            "model_sets": [
                _model_set("Typo", [dict(_spec("x/y"), source="huggingfacce")])
            ]
        }
    )
    with pytest.raises(ValueError, match="Typo"):
        normalize_catalog_yaml(typo, strict=True)

    runner = json.dumps([dict(_entry("1.0.0"), runtime_flavor="new")])
    assert normalize_runner_json(runner)
    with pytest.raises(ValueError, match="runtime_flavor"):
        normalize_runner_json(runner, strict=True)

    backend = yaml.safe_dump(
        [_backend("foo", {"v1": "img:v1"}, telemetry_endpoint="x")]
    )
    assert normalize_backend_yaml(backend)
    with pytest.raises(ValueError, match="telemetry_endpoint"):
        normalize_backend_yaml(backend, strict=True)

    # Inside a version config too, where ``VersionConfig``'s ignored extras hid it.
    in_version = _backend("foo", {"v1": "img:v1"})
    in_version["version_configs"]["v1"]["future_knob"] = "x"
    assert normalize_backend_yaml(yaml.safe_dump([in_version]))
    with pytest.raises(ValueError, match="future_knob"):
        normalize_backend_yaml(yaml.safe_dump([in_version]), strict=True)

    # Nothing is dropped here, but the materialization skips the card.
    unmodelable = yaml.safe_dump(
        [_backend("foo", {"v1": "img:v1"}, parameter_format="future-format")]
    )
    assert normalize_backend_yaml(unmodelable)
    with pytest.raises(ValueError, match="foo"):
        normalize_backend_yaml(unmodelable, strict=True)

    # The check runs on the shape the materialization models: a bare-string
    # built_in_frameworks it normalizes into a list must not come back as a 400.
    bare_framework = yaml.safe_dump(
        [
            {
                "backend_name": "foo",
                "version_configs": {
                    "v1": {"image_name": "img:v1", "built_in_frameworks": "cuda"}
                },
            }
        ]
    )
    assert normalize_backend_yaml(
        bare_framework, strict=True
    ) == normalize_backend_yaml(bare_framework)
