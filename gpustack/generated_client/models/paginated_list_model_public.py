from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.model_public import ModelPublic
    from ..models.pagination import Pagination


T = TypeVar("T", bound="PaginatedListModelPublic")


@_attrs_define
class PaginatedListModelPublic:
    """
    Attributes:
        items (List['ModelPublic']):
        pagination (Pagination):
    """

    items: List["ModelPublic"]
    pagination: "Pagination"
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        items = []
        for items_item_data in self.items:
            items_item = items_item_data.to_dict()
            items.append(items_item)

        pagination = self.pagination.to_dict()

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "items": items,
                "pagination": pagination,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.model_public import ModelPublic
        from ..models.pagination import Pagination

        d = src_dict.copy()
        items = []
        _items = d.pop("items")
        for items_item_data in _items:
            items_item = ModelPublic.from_dict(items_item_data)

            items.append(items_item)

        pagination = Pagination.from_dict(d.pop("pagination"))

        paginated_list_model_public = cls(
            items=items,
            pagination=pagination,
        )

        paginated_list_model_public.additional_properties = d
        return paginated_list_model_public

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
