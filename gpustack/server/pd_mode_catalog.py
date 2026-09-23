import logging
from importlib.resources import files
from typing import Any, Dict, List, Optional

import yaml

from gpustack.schemas.models import PD_MODE_BACKENDS, PDModeEnum
from gpustack.schemas.pd_modes import (
    PDComposedCache,
    PDKVLease,
    PDMode,
    PDModeCatalog,
    PDNetDevicePlaneEnum,
    PDPairingToggle,
    PDTransferMetrics,
    PDRouterProtocolEnum,
    iter_placeholders,
)

logger = logging.getLogger(__name__)

_ASSET_NAME = "pd-modes.yaml"

_catalog: Optional[PDModeCatalog] = None


class PDModeCatalogError(Exception):
    """A malformed or out-of-sync PD-mode catalog.

    Unlike the cache-provider catalog, a broken declaration here is not
    survivable by degrading to an empty catalog: a mode the user can select
    but the catalog cannot answer for is a silent table miss at deploy time
    — nothing injected, and a deployment that comes up looking healthy while
    serving aggregated. So this raises, and the start-up caller lets it fail
    the process.
    """


def load_pd_mode_catalog(reload: bool = False) -> PDModeCatalog:
    """
    Load and validate the declarative PD-mode catalog from the bundled
    asset. The catalog is read-only and cached for the process lifetime, so
    calling this on a request path costs a dict lookup after the first time.
    """
    global _catalog
    if _catalog is not None and not reload:
        return _catalog

    yaml_file = files("gpustack.assets").joinpath(_ASSET_NAME)
    if not yaml_file.is_file():
        raise PDModeCatalogError(f"{_ASSET_NAME} is missing from the installation")
    try:
        raw = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise PDModeCatalogError(f"{_ASSET_NAME} is not valid YAML: {e}") from e

    _catalog = parse_pd_mode_catalog(raw)
    logger.debug(f"Loaded {len(_catalog.modes)} PD modes from {_ASSET_NAME}")
    return _catalog


def parse_pd_mode_catalog(raw: Any) -> PDModeCatalog:
    """
    Parse a catalog document into typed models, resolving each mode's
    kv_lease reference into the connector's declaration, then run the
    cross-source assertions. Split out from the asset read so the contract
    tests can drive it with a document.
    """
    if not isinstance(raw, dict):
        raise PDModeCatalogError(
            f"{_ASSET_NAME} must be a mapping with 'kv_leases' and 'modes' keys"
        )

    leases = _parse_kv_leases(raw.get("kv_leases"))
    composed_cache = _parse_composed_cache(raw.get("composed_cache"))
    pairing_toggles = _parse_pairing_toggles(raw.get("pairing_toggles"))
    transfer_metrics = _parse_transfer_metrics(raw.get("kv_transfer_metrics"), leases)
    modes: List[PDMode] = []
    for entry in raw.get("modes") or []:
        if not isinstance(entry, dict):
            raise PDModeCatalogError(f"mode entries must be mappings, got {entry!r}")
        entry = dict(entry)
        name = entry.get("name")
        reference = entry.get("kv_lease")
        if isinstance(reference, str):
            lease = leases.get(reference)
            if lease is None:
                raise PDModeCatalogError(
                    f"mode '{name}' references kv_lease '{reference}', which is "
                    f"not declared. Declared: {sorted(leases)}"
                )
            entry["kv_lease"] = lease
            # One reference, two resolutions: the connector id decides both
            # the lease window and which counters exist, so a mode never
            # names the transport twice and the two can never disagree.
            entry["transfer_metrics"] = transfer_metrics.get(reference)
        try:
            modes.append(PDMode(**entry))
        except Exception as e:
            raise PDModeCatalogError(f"mode '{name}' is invalid: {e}") from e

    _assert_names_match_enum(modes)
    _assert_backends_match_table(modes)
    _assert_expired_metric_agrees(modes)
    _assert_gpu_filters_declared(modes)
    _assert_router_invocation_is_classified(modes)
    _assert_net_device_plane_is_read_somewhere(modes)
    _assert_composed_cache_backends_are_declared(composed_cache, modes)
    _assert_pairing_toggle_backends_are_declared(pairing_toggles, modes)
    return PDModeCatalog(
        kv_leases=leases,
        composed_cache=composed_cache,
        pairing_toggles=pairing_toggles,
        kv_transfer_metrics=transfer_metrics,
        modes=modes,
    )


def _parse_pairing_toggles(raw: Any) -> List[PDPairingToggle]:
    """Parse the boolean engine switches both roles must agree on."""
    toggles: List[PDPairingToggle] = []
    seen = set()
    for entry in raw or []:
        if not isinstance(entry, dict):
            raise PDModeCatalogError(
                f"pairing_toggles entries must be mappings, got {entry!r}"
            )
        try:
            declared = PDPairingToggle(**entry)
        except Exception as e:
            raise PDModeCatalogError(
                f"invalid pairing_toggles entry {entry!r}: {e}"
            ) from e
        identity = (declared.backend, declared.key)
        if identity in seen:
            raise PDModeCatalogError(
                f"duplicate pairing_toggle '{declared.key}' for backend "
                f"'{declared.backend}'"
            )
        seen.add(identity)
        toggles.append(declared)
    return toggles


def _assert_pairing_toggle_backends_are_declared(
    toggles: List[PDPairingToggle], modes: List[PDMode]
) -> None:
    """Same reason as the composed-cache check: the lookup keys on the backend
    string, so a misspelling stops applying in silence."""
    known = {backend for mode in modes for backend in (mode.backends or [])}
    unknown = sorted({toggle.backend for toggle in toggles} - known)
    if unknown:
        raise PDModeCatalogError(
            f"pairing_toggles names backends no mode declares: {unknown}. "
            f"Declared by modes: {sorted(known)}"
        )


def _parse_composed_cache(raw: Any) -> Dict[str, PDComposedCache]:
    """Parse the per-backend requirements for folding a cache in beside the PD
    connector. A backend with no entry composes unchecked, deliberately."""
    composed: Dict[str, PDComposedCache] = {}
    for entry in raw or []:
        if not isinstance(entry, dict):
            raise PDModeCatalogError(
                f"composed_cache entries must be mappings, got {entry!r}"
            )
        try:
            declared = PDComposedCache(**entry)
        except Exception as e:
            raise PDModeCatalogError(
                f"invalid composed_cache entry {entry!r}: {e}"
            ) from e
        if declared.backend in composed:
            raise PDModeCatalogError(
                f"duplicate composed_cache for backend '{declared.backend}'"
            )
        composed[declared.backend] = declared
    return composed


def _assert_composed_cache_backends_are_declared(
    composed: Dict[str, PDComposedCache], modes: List[PDMode]
) -> None:
    """A requirement for a backend no mode runs on is enforced against nothing.

    Catches the spelling slip the runtime would otherwise swallow: the check
    keys on the backend string, so 'vllm' instead of 'vLLM' silently stops
    applying rather than failing.
    """
    known = {backend for mode in modes for backend in (mode.backends or [])}
    unknown = sorted(set(composed) - known)
    if unknown:
        raise PDModeCatalogError(
            f"composed_cache names backends no mode declares: {unknown}. "
            f"Declared by modes: {sorted(known)}"
        )
    _assert_refused_roles_exist(composed, modes)


def _assert_refused_roles_exist(
    composed: Dict[str, PDComposedCache], modes: List[PDMode]
) -> None:
    """A refused role no recipe declares refuses nothing.

    Same failure shape as the backend check above, one level down: the refusal
    is matched against the member's role name, so 'decoder' instead of 'decode'
    would let a cache through onto the side that dies on a hit — and nothing at
    runtime would say so.
    """
    for backend, declared in composed.items():
        refusal = declared.refuse_cache_on
        if refusal is None or not refusal.roles:
            continue
        known = {
            name
            for mode in modes
            if backend in (mode.backends or [])
            for name in (mode.roles or {})
        }
        unknown = sorted(set(refusal.roles) - known)
        if unknown:
            raise PDModeCatalogError(
                f"composed_cache for '{backend}' refuses roles no recipe of "
                f"that backend declares: {unknown}. Declared: {sorted(known)}"
            )


def _parse_kv_leases(raw: Any) -> Dict[str, PDKVLease]:
    leases: Dict[str, PDKVLease] = {}
    for entry in raw or []:
        if not isinstance(entry, dict):
            raise PDModeCatalogError(
                f"kv_lease entries must be mappings, got {entry!r}"
            )
        try:
            lease = PDKVLease(**entry)
        except Exception as e:
            raise PDModeCatalogError(f"invalid kv_lease entry {entry!r}: {e}") from e
        if lease.connector in leases:
            raise PDModeCatalogError(f"duplicate kv_lease for '{lease.connector}'")
        leases[lease.connector] = lease
    return leases


def _parse_transfer_metrics(
    raw: Any, leases: Dict[str, PDKVLease]
) -> Dict[str, PDTransferMetrics]:
    """Parse the per-connector transfer-counter registry.

    Every connector with a lease must have an entry, even an all-null one.
    "Mooncake exports no Prometheus counters" is a measured fact and has to
    be written down; a connector simply missing from this table produces
    the same runtime behaviour — no numerator — from an oversight, and the
    two must not be indistinguishable.
    """
    metrics: Dict[str, PDTransferMetrics] = {}
    for entry in raw or []:
        if not isinstance(entry, dict):
            raise PDModeCatalogError(
                f"kv_transfer_metrics entries must be mappings, got {entry!r}"
            )
        try:
            declared = PDTransferMetrics(**entry)
        except Exception as e:
            raise PDModeCatalogError(
                f"invalid kv_transfer_metrics entry {entry!r}: {e}"
            ) from e
        if declared.connector in metrics:
            raise PDModeCatalogError(
                f"duplicate kv_transfer_metrics for '{declared.connector}'"
            )
        metrics[declared.connector] = declared

    undeclared = sorted(set(leases) - set(metrics))
    if undeclared:
        raise PDModeCatalogError(
            "every connector with a kv_lease must also declare its transfer "
            "counters, all-null if it exports none, so an absence is a "
            f"statement rather than an omission. Missing: {undeclared}"
        )
    unknown = sorted(set(metrics) - set(leases))
    if unknown:
        raise PDModeCatalogError(
            f"kv_transfer_metrics names connectors no kv_lease declares: {unknown}"
        )
    return metrics


def _assert_names_match_enum(modes: List[PDMode]) -> None:
    """The catalog's entry names must be exactly PDModeEnum's values.

    A name present on only one side does not raise anywhere near where it
    was written: the request validates against the enum, the injector then
    misses the table and injects nothing, and the deployment serves
    aggregated with no error anywhere. This has happened once already
    (`ascend-mooncake` vs `vllm-ascend-mooncake`), so it fails start-up
    instead.
    """
    declared = {mode.name for mode in modes}
    expected = {mode.value for mode in PDModeEnum}
    if declared == expected:
        return
    missing = sorted(expected - declared)
    unknown = sorted(declared - expected)
    raise PDModeCatalogError(
        f"{_ASSET_NAME} and PDModeEnum disagree — the catalog is looked up by "
        "mode name, so a mismatch is a silent table miss, not an error: "
        f"in the enum but missing from the catalog: {missing}; "
        f"in the catalog but not in the enum: {unknown}"
    )


def _assert_backends_match_table(modes: List[PDMode]) -> None:
    """Each entry's `backends` must agree with PD_MODE_BACKENDS.

    That table exists only so request validation does not have to read the
    catalog. This catalog is the authoritative declaration of which engines
    a recipe may be injected into, and this assertion is what keeps the copy
    honest — otherwise a mode added here could pass a validation the catalog
    itself would refuse.
    """
    disagreements = []
    for mode in modes:
        declared = sorted(mode.backends)
        expected = sorted(PD_MODE_BACKENDS.get(mode.name, []))
        if declared != expected:
            disagreements.append(
                f"'{mode.name}': catalog {declared} != table {expected}"
            )
    if disagreements:
        raise PDModeCatalogError(
            "PD_MODE_BACKENDS is a copy of this catalog's `backends` kept for "
            "request validation, and the two disagree: " + "; ".join(disagreements)
        )


def _assert_router_invocation_is_classified(modes: List[PDMode]) -> None:
    """A shipped router declares its invocation in the three classified parts.

    ``PDRouter`` accepts a bare ``command`` too — that is what keeps the type
    usable outside the catalog — so the requirement that *our* recipes classify
    theirs belongs here, where the subject is what we ship.

    Two things depend on the classification, and both fail silently without it:

    - The deploy form cannot tell a user which router flags they may change.
      Falling back to "the whole command, read-only" is a usable degradation,
      so this alone would not justify raising.
    - **The refusal list is read off ``connection_args``.** An unclassified
      router has an empty one, which turns "you may not set ``--prefill``"
      into "you may", and a second ``--prefill`` does not replace the injected
      peer — both shipped routers declare it ``action="append"``, so it adds
      one the router forwards to and cannot reach. That is the half that has
      to be caught at load time rather than at deploy time.
    """
    problems = []
    for mode in modes:
        router = mode.router
        if router is None:
            continue
        if router.protocol == PDRouterProtocolEnum.USER_PROVIDED:
            continue
        if not router.entrypoint:
            problems.append(
                f"'{mode.name}': router declares no entrypoint — say which "
                f"executable inside the image runs"
            )
        if not router.connection_args:
            problems.append(
                f"'{mode.name}': router declares no connection_args — the "
                f"flags a deployment may not override are read from there, so "
                f"an empty list silently permits all of them"
            )
    if problems:
        raise PDModeCatalogError("; ".join(problems))


def _assert_net_device_plane_is_read_somewhere(modes: List[PDMode]) -> None:
    """A recipe that declares `net_device_plane: control` must reference
    `{{net_device}}`.

    The field only ever changes the value of that one placeholder, so declaring
    it on a recipe that never injects the placeholder is a statement that does
    nothing — and the shape it would be written in by mistake is the expensive
    one: somebody relaxing the multi-NIC refusal for `sglang-mooncake`, whose
    NIC is the engine's own business (`--disaggregation-ib-device`), would see
    no effect and go looking for the reason in code.

    Only the non-default direction is checked. `data` is what a recipe gets for
    saying nothing, so requiring it to be *used* would refuse every recipe that
    legitimately leaves the NIC to the engine.
    """
    problems = []
    for mode in modes:
        if mode.net_device_plane is not PDNetDevicePlaneEnum.CONTROL:
            continue
        referenced = any(
            occurrence == "{{net_device}}" for occurrence in iter_placeholders(mode)
        )
        if not referenced:
            problems.append(
                f"'{mode.name}' declares net_device_plane "
                f"'{mode.net_device_plane.value}' but injects no "
                "{{net_device}} anywhere, so the declaration changes nothing"
            )
    if problems:
        raise PDModeCatalogError("; ".join(problems))


def _assert_gpu_filters_declared(modes: List[PDMode]) -> None:
    """Every recipe that injects something must say which accelerators it fits;
    the one that injects nothing must not.

    Both halves are load-bearing and neither is obvious:

    - **A built-in recipe without `gpu_filters` is offered everywhere.** That
      is how three NVIDIA-only recipes came to be selectable on Ascend and on
      AMD -- the earlier `runtime` field left "unconstrained" and "not
      declared" spelled the same way, so forgetting the constraint looked
      exactly like meaning "any accelerator".
    - **`custom` must stay unconstrained.** It injects nothing, so it is the
      only way to run PD on an engine × accelerator pair we ship no recipe
      for. Giving it a filter would turn "no built-in recipe" into "no PD".

    Keyed off `roles`/`router` rather than a name list so a fourth built-in
    recipe is caught by the same rule instead of needing an edit here.
    """
    problems = []
    for mode in modes:
        injects = bool(mode.roles) or (
            mode.router is not None
            and mode.router.protocol is not PDRouterProtocolEnum.USER_PROVIDED
        )
        declared = mode.gpu_filters is not None and bool(mode.gpu_filters.vendor)
        if injects and not declared:
            problems.append(
                f"'{mode.name}' injects configuration but declares no "
                "gpu_filters.vendor, so it would be offered on every "
                "accelerator"
            )
        if injects and not mode.transport:
            problems.append(
                f"'{mode.name}' injects configuration but names no "
                "transport, so the derived one-liner has nothing to show"
            )
        if not injects and mode.transport:
            problems.append(
                f"'{mode.name}' injects nothing, so it has no transport of "
                "its own to name"
            )
        if not injects and declared:
            problems.append(
                f"'{mode.name}' injects nothing, so a gpu_filters constraint "
                "would only remove the escape hatch for accelerators we ship "
                "no recipe for"
            )
    if problems:
        raise PDModeCatalogError("; ".join(problems))


def _assert_expired_metric_agrees(modes: List[PDMode]) -> None:
    """`router.capabilities.kv_expired_metric` is the boolean the metrics
    collector reads; the lease registry carries the metric's name. Two
    spellings of one fact, so they have to agree — Mooncake exports no
    counter, and a mode claiming otherwise would poll for a metric that
    never appears."""
    for mode in modes:
        if mode.router is None:
            continue
        claimed = mode.router.capabilities.kv_expired_metric
        available = mode.kv_lease is not None and bool(mode.kv_lease.expired_metric)
        if claimed != available:
            declared = mode.kv_lease.expired_metric if mode.kv_lease else None
            raise PDModeCatalogError(
                f"mode '{mode.name}' claims kv_expired_metric={claimed} but its "
                f"kv_lease declares expired_metric={declared}"
            )


def load_pd_modes(reload: bool = False) -> List[PDMode]:
    return load_pd_mode_catalog(reload=reload).modes


def get_pd_modes() -> List[PDMode]:
    return load_pd_mode_catalog().modes


def get_pd_mode(name: str) -> Optional[PDMode]:
    return load_pd_mode_catalog().mode(name)


def get_kv_leases() -> Dict[str, PDKVLease]:
    """Every connector's lease window, including connectors no shipped mode
    uses (MoRIIO) — which is the point of the whole-registry accessor, since a
    mode's own window is already resolved onto the mode."""
    return load_pd_mode_catalog().kv_leases


def get_kv_lease(connector: str) -> Optional[PDKVLease]:
    return load_pd_mode_catalog().kv_leases.get(connector)


def get_transfer_metrics(connector: str) -> Optional[PDTransferMetrics]:
    """One connector's KV-transfer counters. A mode's own are already
    resolved onto the mode; this is for the connector-first callers."""
    return load_pd_mode_catalog().kv_transfer_metrics.get(connector)


def get_pairing_toggles(backend: str) -> List[PDPairingToggle]:
    """Boolean engine switches both roles of a `backend` pair must agree on."""
    return [
        toggle
        for toggle in load_pd_mode_catalog().pairing_toggles
        if toggle.backend == backend
    ]


def get_composed_cache(backend: str) -> Optional[PDComposedCache]:
    """What this backend requires before a shared cache is folded in beside
    the PD connector, or None when it declares nothing — which is the answer
    for an engine that never composes the two."""
    return load_pd_mode_catalog().composed_cache.get(backend)
