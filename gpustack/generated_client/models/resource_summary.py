from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.resource_summary_allocatable import ResourceSummaryAllocatable
    from ..models.resource_summary_capacity import ResourceSummaryCapacity


T = TypeVar("T", bound="ResourceSummary")


@_attrs_define
class ResourceSummary:
    """
    Attributes:
        capacity (Union[Unset, ResourceSummaryCapacity]):
        allocatable (Union[Unset, ResourceSummaryAllocatable]):
    """

    capacity: Union[Unset, "ResourceSummaryCapacity"] = UNSET
    allocatable: Union[Unset, "ResourceSummaryAllocatable"] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        capacity: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.capacity, Unset):
            capacity = self.capacity.to_dict()

        allocatable: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.allocatable, Unset):
            allocatable = self.allocatable.to_dict()

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if capacity is not UNSET:
            field_dict["capacity"] = capacity
        if allocatable is not UNSET:
            field_dict["allocatable"] = allocatable

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.resource_summary_allocatable import ResourceSummaryAllocatable
        from ..models.resource_summary_capacity import ResourceSummaryCapacity

        d = src_dict.copy()
        _capacity = d.pop("capacity", UNSET)
        capacity: Union[Unset, ResourceSummaryCapacity]
        if isinstance(_capacity, Unset):
            capacity = UNSET
        else:
            capacity = ResourceSummaryCapacity.from_dict(_capacity)

        _allocatable = d.pop("allocatable", UNSET)
        allocatable: Union[Unset, ResourceSummaryAllocatable]
        if isinstance(_allocatable, Unset):
            allocatable = UNSET
        else:
            allocatable = ResourceSummaryAllocatable.from_dict(_allocatable)

        resource_summary = cls(
            capacity=capacity,
            allocatable=allocatable,
        )

        resource_summary.additional_properties = d
        return resource_summary

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
