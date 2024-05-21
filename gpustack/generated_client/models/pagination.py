from typing import Any, Dict, List, Type, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="Pagination")


@_attrs_define
class Pagination:
    """
    Attributes:
        page (int):
        per_page (int):
        total (int):
        total_page (int):
    """

    page: int
    per_page: int
    total: int
    total_page: int
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        page = self.page

        per_page = self.per_page

        total = self.total

        total_page = self.total_page

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "page": page,
                "perPage": per_page,
                "total": total,
                "totalPage": total_page,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        page = d.pop("page")

        per_page = d.pop("perPage")

        total = d.pop("total")

        total_page = d.pop("totalPage")

        pagination = cls(
            page=page,
            per_page=per_page,
            total=total,
            total_page=total_page,
        )

        pagination.additional_properties = d
        return pagination

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
