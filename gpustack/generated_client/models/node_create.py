from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.node_create_labels import NodeCreateLabels
    from ..models.resource_summary import ResourceSummary


T = TypeVar("T", bound="NodeCreate")


@_attrs_define
class NodeCreate:
    """
    Attributes:
        name (str):
        hostname (str):
        address (str):
        resources (Union['ResourceSummary', None]):
        labels (Union[Unset, NodeCreateLabels]):
        state (Union[None, Unset, str]):
    """

    name: str
    hostname: str
    address: str
    resources: Union["ResourceSummary", None]
    labels: Union[Unset, "NodeCreateLabels"] = UNSET
    state: Union[None, Unset, str] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        from ..models.resource_summary import ResourceSummary

        name = self.name

        hostname = self.hostname

        address = self.address

        resources: Union[Dict[str, Any], None]
        if isinstance(self.resources, ResourceSummary):
            resources = self.resources.to_dict()
        else:
            resources = self.resources

        labels: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.labels, Unset):
            labels = self.labels.to_dict()

        state: Union[None, Unset, str]
        if isinstance(self.state, Unset):
            state = UNSET
        else:
            state = self.state

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "hostname": hostname,
                "address": address,
                "resources": resources,
            }
        )
        if labels is not UNSET:
            field_dict["labels"] = labels
        if state is not UNSET:
            field_dict["state"] = state

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.node_create_labels import NodeCreateLabels
        from ..models.resource_summary import ResourceSummary

        d = src_dict.copy()
        name = d.pop("name")

        hostname = d.pop("hostname")

        address = d.pop("address")

        def _parse_resources(data: object) -> Union["ResourceSummary", None]:
            if data is None:
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                resources_type_0 = ResourceSummary.from_dict(data)

                return resources_type_0
            except:  # noqa: E722
                pass
            return cast(Union["ResourceSummary", None], data)

        resources = _parse_resources(d.pop("resources"))

        _labels = d.pop("labels", UNSET)
        labels: Union[Unset, NodeCreateLabels]
        if isinstance(_labels, Unset):
            labels = UNSET
        else:
            labels = NodeCreateLabels.from_dict(_labels)

        def _parse_state(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        state = _parse_state(d.pop("state", UNSET))

        node_create = cls(
            name=name,
            hostname=hostname,
            address=address,
            resources=resources,
            labels=labels,
            state=state,
        )

        node_create.additional_properties = d
        return node_create

    @property
    def additional_keys(self) -> List[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
