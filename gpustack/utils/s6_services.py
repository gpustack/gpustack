from dataclasses import dataclass
from typing import Optional, Set, List, Dict

gpustack_service_name = "gpustack"


@dataclass
class S6Service:
    name: str
    ports: Optional[List[int | str]] = None
    is_dependency: bool = False
    longrun: bool = True


class S6Services:
    services: Set[str]
    support_pipeline: bool = False
    pipeline_prefix: str = "pipeline-"
    dependencies: Set[str]  # dependency services
    service_port_getters: Dict[
        str, List[str | int]
    ]  # service name to port or config field

    def __init__(
        self,
        *services: S6Service,
        support_pipeline: bool = False,
        pipeline_prefix: str = "pipeline-",
    ):
        self.services = set()
        self.dependencies = set()
        self.service_port_getters = {}
        self.support_pipeline = support_pipeline
        self.pipeline_prefix = pipeline_prefix
        for service in services:
            if service.longrun:
                self.services.add(service.name)
            if service.is_dependency:
                self.dependencies.add(service.name)
            if service.ports:
                self.service_port_getters[service.name] = list(service.ports)

    def all_services(self) -> List[str]:
        if self.support_pipeline:
            pipeline_services = [
                self.pipeline_prefix + service for service in self.services
            ]
            return list(self.services) + pipeline_services
        return list(self.services)

    def set_ports(self, config: object, ports: Dict[int, str]):
        if not self.service_port_getters:
            return
        for service, port_list in self.service_port_getters.items():
            for port_or_field in port_list:
                if isinstance(port_or_field, int):
                    ports[port_or_field] = service
                else:
                    port_value = getattr(config, port_or_field, None)
                    if port_value is None or not isinstance(port_value, int):
                        continue
                    if not port_conflict(port_value, ports):
                        ports[port_value] = service

    @property
    def dep_services(self) -> List[str]:
        return list(self.dependencies or [])


gateway_services = S6Services(
    S6Service("apiserver", [18443], True),
    S6Service("pilot", [9876, 15010, 15012]),
    S6Service("controller", [8888, 15051]),
    S6Service("gateway", [15000, 15021, 15090, 15020]),
    S6Service("supercronic"),
    support_pipeline=True,
)
postgres_services = S6Services(
    S6Service("postgres", ["database_port"], True),
)
migration_services = S6Services(
    S6Service("gpustack-migration", [], True, False),
)
observability_services = S6Services(
    S6Service("grafana", [3000]),
    S6Service("prometheus", [9090]),
    support_pipeline=True,
)


def all_dependent_services() -> List[str]:
    return [
        *gateway_services.dep_services,
        *postgres_services.dep_services,
        *migration_services.dep_services,
        *observability_services.dep_services,
    ]


def all_services() -> List[str]:
    return [
        *gateway_services.all_services(),
        *postgres_services.all_services(),
        *migration_services.all_services(),
        *observability_services.all_services(),
    ]


def port_conflict(port: int, ports: Dict[int, str]) -> bool:
    existing_service = ports.get(port, None)
    if existing_service is not None:
        raise Exception(f"Port conflict: {port} is already used by " + existing_service)
    return False
