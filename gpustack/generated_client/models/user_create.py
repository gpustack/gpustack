from typing import Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="UserCreate")


@_attrs_define
class UserCreate:
    """
    Attributes:
        name (str):
        password (str):
        is_admin (Union[Unset, bool]):  Default: False.
        full_name (Union[None, Unset, str]):
    """

    name: str
    password: str
    is_admin: Union[Unset, bool] = False
    full_name: Union[None, Unset, str] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        name = self.name

        password = self.password

        is_admin = self.is_admin

        full_name: Union[None, Unset, str]
        if isinstance(self.full_name, Unset):
            full_name = UNSET
        else:
            full_name = self.full_name

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "password": password,
            }
        )
        if is_admin is not UNSET:
            field_dict["is_admin"] = is_admin
        if full_name is not UNSET:
            field_dict["full_name"] = full_name

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        name = d.pop("name")

        password = d.pop("password")

        is_admin = d.pop("is_admin", UNSET)

        def _parse_full_name(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        full_name = _parse_full_name(d.pop("full_name", UNSET))

        user_create = cls(
            name=name,
            password=password,
            is_admin=is_admin,
            full_name=full_name,
        )

        user_create.additional_properties = d
        return user_create

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
