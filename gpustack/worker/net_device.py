import ipaddress
import logging
import socket
from typing import List, Optional

import psutil

from gpustack.config.config import Config
from gpustack.schemas.pd_modes import PDNetDevicePlaneEnum
from gpustack.schemas.workers import Worker

logger = logging.getLogger(__name__)


# Interfaces that exist only to move packets between namespaces on this host.
# The KV plane can never ride them, and letting one through fails like this:
# UCX advertises the address behind the bridge (``172.17.x``,
# ``10.42.x``) in the NIXL metadata, the peer cannot route to it, and
# ``loadRemoteMD()`` reports ``NIXL_ERR_BACKEND`` a whole handshake away from
# the machine that was misconfigured.
#
# Matched as name prefixes. The list is deliberately generous, because the cost
# of the two mistakes is not symmetric: a virtual device left in the candidate
# set inflates the count and makes a perfectly ordinary single-NIC host with
# Docker installed refuse to derive anything, while a real NIC wrongly excluded
# only shrinks a set that is used to *gate* a decision, never to make one.
_VIRTUAL_IFNAME_PREFIXES = (
    # container runtimes / CNIs
    "docker",
    "br-",  # docker user-defined bridges are br-<hex>; a hand-made br0 is
    # frequently the host's real uplink, so bare "br" is not excluded
    "veth",
    "cni",
    "flannel",
    "cali",
    "cilium_",
    "lxc",
    "weave",
    "datapath",
    "antrea",
    "kube-",
    "nodelocaldns",
    "vxlan",
    "genev_sys",
    "ovs-system",
    # hypervisors and tunnels
    "virbr",
    "vnet",
    "tunl",
    "tun",
    "tap",
    "wg",
    "ppp",
    "tailscale",
    "zt",
    "dummy",
    "utun",  # macOS
    "bridge",  # macOS (bridge100 = the Internet Sharing / VM bridge)
)


def _is_virtual_ifname(name: str) -> bool:
    return name.startswith(_VIRTUAL_IFNAME_PREFIXES)


def _has_routable_address(addrs) -> bool:
    """Whether the interface carries an address a peer could connect back to.

    An interface with only a link-local or loopback address contributes nothing
    to the NIXL metadata, so it is not a plausible KV NIC -- this is what keeps
    the dozens of ``veth`` stubs and the unplugged onboard ports out of the
    count even when the name filter above does not recognise them.
    """
    for addr in addrs:
        if addr.family not in (socket.AF_INET, socket.AF_INET6):
            continue
        raw = (addr.address or "").split("%", 1)[0]  # strip the IPv6 zone index
        try:
            ip = ipaddress.ip_address(raw)
        except ValueError:
            continue
        if ip.is_loopback or ip.is_link_local or ip.is_unspecified:
            continue
        return True
    return False


def candidate_kv_interfaces() -> List[str]:
    """The host NICs that could plausibly carry the KV plane.

    Runs in the worker process, which is the only place the question can be
    answered at all: ``WorkerStatus`` carries no host NIC inventory, so the
    server cannot count the interfaces on a machine it is not running on.

    The result is used as a *gate*, not as a source of values -- see
    ``derive_net_device``.
    """
    stats = psutil.net_if_stats()
    candidates = []
    for name, addrs in psutil.net_if_addrs().items():
        if _is_virtual_ifname(name):
            continue
        stat = stats.get(name)
        if stat is not None:
            # A down NIC cannot carry KV traffic, and "loopback" is a flag
            # rather than a naming convention on every platform.
            if not stat.isup or "loopback" in (stat.flags or ""):
                continue
        if not _has_routable_address(addrs):
            continue
        candidates.append(name)
    return sorted(candidates)


def _multi_nic_refusal(worker_name: str, worker_ifname: str, candidates: List[str]):
    detail = (
        f" The detected management-plane NIC is {worker_ifname!r}; set "
        f"kv_ifname to it if that is really where the KV traffic belongs."
        if worker_ifname
        else ""
    )
    logger.error(
        "Refusing to derive the KV plane network interface for worker %s: "
        "this host has %d candidate interfaces (%s) and nothing here can tell "
        "which one the KV traffic should ride. Set kv_ifname on this worker to "
        "name it explicitly.%s",
        worker_name,
        len(candidates),
        ", ".join(candidates),
        detail,
    )


def derive_net_device(
    worker: Worker,
    config: Config,
    plane: PDNetDevicePlaneEnum = PDNetDevicePlaneEnum.DATA,
) -> Optional[str]:
    """The value of ``{{net_device}}``: the NIC this recipe's ``plane`` rides.

    Priority: ``config.kv_ifname`` (per-worker escape hatch) -> ``Worker.ifname``
    *if this host has only one plausible KV NIC* -> ``None``.

    **``plane`` decides whether that middle gate applies at all**, and it is
    passed in from the recipe (``PDMode.net_device_plane``) rather than decided
    here, because one placeholder is injected into variables of two natures —
    see ``PDNetDevicePlaneEnum``. On ``CONTROL`` the host's NIC list is not
    consulted: the value wanted there is the NIC holding the worker's
    registration IP, which is what ``Worker.ifname`` *is*, so there is nothing
    to guess between and a refusal would only make the operator retype a fact
    the platform already knows — or mistype it, which on Ascend puts
    ``HCCL_IF_IP`` on a NIC that does not hold it.

    ``kv_ifname`` still wins on both planes. It is the per-worker escape hatch,
    and a plane that could bypass it would make setting it a no-op exactly
    where an operator went to the trouble of answering.

    Do NOT invent a value when this returns ``None``. The renderer leaves an
    unresolved placeholder as-is and logs a WARNING, which is diagnosable;
    falling back to ``all`` is not. UCX with ``UCX_NET_DEVICES=all`` picks up
    ``docker0`` / ``br-*`` / ``flannel.1`` / ``cni0`` and writes the addresses
    behind them (``172.17.x``, ``10.42.x``) into the NIXL metadata. The peer
    cannot route to those, so ``loadRemoteMD()`` fails with
    ``NIXL_ERR_BACKEND`` -- a wrong value instead of a clean failure, at the
    far end of the handshake rather than at the point of the misconfiguration.

    **The multi-NIC refusal (data plane only).** ``Worker.ifname`` is the
    management-plane NIC. On a single-NIC host that is the right answer; on a
    multi-NIC host, or when the KV traffic is meant to ride a dedicated fabric,
    it is not, and only the operator knows which. So the host's own NIC list is
    enumerated here and used purely as a gate: more than one candidate means the
    fallback is a guess, and a guess is refused with an ERROR naming the
    candidates rather than silently resolved to the management plane.

    The enumeration never *supplies* a value, only withholds one. Picking "the
    single candidate" over ``Worker.ifname`` would swap a measured fact -- the
    NIC that demonstrably carries worker->server traffic -- for an inference
    drawn from a hand-written pattern list, which is the same class of guess.

    **Why ``None`` and not an exception.** The one caller
    (``ModelInstanceBackend._pd_template_variables``) already funnels every
    exception into a WARNING and drops the variable, so raising would not stop
    a start -- it would only lose control of the log level. Returning ``None``
    keeps the refusal at ERROR and leaves ``{{net_device}}`` unresolved, which
    fails this instance's engine at launch with the placeholder still visible in
    its command line. That blast radius is exactly right: ``derive_net_device``
    is reached only from the PD/managed-router path (``_pd_injection`` returns
    early unless ``model.disaggregation`` is set), so an ordinary model on the
    same multi-NIC host is untouched.

    Per-card interfaces are deliberately *not* consulted here, even though
    ``GPUDeviceStatus.network.iface`` exists:

    - On Ascend the detector reads it out of ``hccn_tool``, where it names the
      card-internal ``eth0``-``eth7``. Those devices do not exist in the host
      network namespace, so UCX cannot bind to them -- it is a card identifier,
      not a NIC name.
    - On other vendors the field is only ever populated by hand via
      ``resources.gpu_devices``, and filling that in swaps the whole GPU
      detector for the ``Custom`` one, freezing VRAM/model/power discovery.
      That is a worse trade than typing one ``kv_ifname``.

    One boundary remains, and it lands the caller on ``kv_ifname`` too: when
    ``gpu_type_selector`` is set (the only way to get a gang), the cards are
    assigned by the device plugin *after* the pod binds, so no per-card source
    could have been used at render time anyway.
    """
    # ``Worker.ifname`` is a non-optional ``str``, but pool-provisioned rows are
    # created with "" as a placeholder before the worker ever reports in, so
    # emptiness -- not absence -- is what marks "unknown" on both sources.
    kv_ifname = (config.kv_ifname or "").strip()
    if kv_ifname:
        return kv_ifname

    worker_name = worker.name or "<unknown>"
    worker_ifname = (worker.ifname or "").strip()

    # The control plane asks a question the host's NIC list cannot answer and
    # does not need to: the NIC wanted there is the one carrying the worker's
    # registration IP, and nothing in an enumeration identifies that. So the
    # gate below is skipped entirely rather than widened -- an empty candidate
    # set also keeps the "looks virtual" warning silent, which on this plane
    # would be wrong twice over: a worker genuinely reaching the server through
    # such an interface is describing the route its own handshake must take.
    candidates: List[str] = []
    if plane is not PDNetDevicePlaneEnum.CONTROL:
        try:
            candidates = candidate_kv_interfaces()
        except Exception as e:
            # A failed enumeration must not be able to withhold a value on its
            # own: the gate exists to stop a guess, and treating "I could not
            # look" as "there are too many" would break hosts that work today.
            logger.warning(
                "Failed to enumerate the network interfaces of worker %s (%s); "
                "falling back to the management-plane NIC without the multi-NIC "
                "check.",
                worker_name,
                e,
            )
            candidates = []

        if len(candidates) > 1:
            _multi_nic_refusal(worker_name, worker_ifname, candidates)
            return None

    if worker_ifname:
        if candidates and worker_ifname not in candidates:
            # Warned, not refused. This is the one shape where the measured NIC
            # is known-bad (a worker that reaches the server through docker0
            # reports docker0), but the judgement rests entirely on the
            # hand-written prefix list above, and a false positive there must
            # not be able to break a deployment that works today.
            logger.warning(
                "The management-plane NIC %r of worker %s is not among the "
                "candidate KV interfaces (%s) -- it looks virtual or down. "
                "UCX may advertise an address the peer cannot route; set "
                "kv_ifname if the KV plane belongs elsewhere.",
                worker_ifname,
                worker_name,
                ", ".join(candidates),
            )
        return worker_ifname

    logger.warning(
        "Cannot derive the KV plane network interface for worker %s: "
        "neither kv_ifname nor a detected worker ifname is available. "
        "Set kv_ifname on that worker to name the interface explicitly.",
        worker_name,
    )
    return None
