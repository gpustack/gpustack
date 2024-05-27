import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.node_public_labels import NodePublicLabels
    from ..models.resource_summary import ResourceSummary


T = TypeVar("T", bound="NodePublic")


@_attrs_define
class NodePublic:
    """
    Attributes:
        name (str):
        hostname (str):
        address (str):
        resources (Union['ResourceSummary', None]):
        id (int):
        created_at (datetime.datetime):
        updated_at (datetime.datetime):
        labels (Union[Unset, NodePublicLabels]):
        state (Union[None, Unset, str]):
    """

    name: str
    hostname: str
    address: str
    resources: Union["ResourceSummary", None]
    id: int
    created_at: datetime.datetime
    updated_at: datetime.datetime
    labels: Union[Unset, "NodePublicLabels"] = UNSET
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

        id = self.id

        created_at = self.created_at.isoformat()

        updated_at = self.updated_at.isoformat()

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
                "id": id,
                "created_at": created_at,
                "updated_at": updated_at,
            }
        )
        if labels is not UNSET:
            field_dict["labels"] = labels
        if state is not UNSET:
            field_dict["state"] = state

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.node_public_labels import NodePublicLabels
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

        id = d.pop("id")

        created_at = isoparse(d.pop("created_at"))

        updated_at = isoparse(d.pop("updated_at"))

        _labels = d.pop("labels", UNSET)
        labels: Union[Unset, NodePublicLabels]
        if isinstance(_labels, Unset):
            labels = UNSET
        else:
            labels = NodePublicLabels.from_dict(_labels)

        def _parse_state(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        state = _parse_state(d.pop("state", UNSET))

        node_public = cls(
            name=name,
            hostname=hostname,
            address=address,
            resources=resources,
            id=id,
            created_at=created_at,
            updated_at=updated_at,
            labels=labels,
            state=state,
        )

        node_public.additional_properties = d
        return node_public

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
