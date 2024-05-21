from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

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
        id (str):
        name (str):
        hostname (str):
        address (str):
        resources (ResourceSummary):
        state (str):
        labels (Union[Unset, NodeCreateLabels]):
    """

    id: str
    name: str
    hostname: str
    address: str
    resources: "ResourceSummary"
    state: str
    labels: Union[Unset, "NodeCreateLabels"] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        id = self.id

        name = self.name

        hostname = self.hostname

        address = self.address

        resources = self.resources.to_dict()

        state = self.state

        labels: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.labels, Unset):
            labels = self.labels.to_dict()

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "name": name,
                "hostname": hostname,
                "address": address,
                "resources": resources,
                "state": state,
            }
        )
        if labels is not UNSET:
            field_dict["labels"] = labels

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.node_create_labels import NodeCreateLabels
        from ..models.resource_summary import ResourceSummary

        d = src_dict.copy()
        id = d.pop("id")

        name = d.pop("name")

        hostname = d.pop("hostname")

        address = d.pop("address")

        resources = ResourceSummary.from_dict(d.pop("resources"))

        state = d.pop("state")

        _labels = d.pop("labels", UNSET)
        labels: Union[Unset, NodeCreateLabels]
        if isinstance(_labels, Unset):
            labels = UNSET
        else:
            labels = NodeCreateLabels.from_dict(_labels)

        node_create = cls(
            id=id,
            name=name,
            hostname=hostname,
            address=address,
            resources=resources,
            state=state,
            labels=labels,
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
