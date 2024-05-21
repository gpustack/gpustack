from typing import Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.source_enum import SourceEnum
from ..types import UNSET, Unset

T = TypeVar("T", bound="ModelUpdate")


@_attrs_define
class ModelUpdate:
    """
    Attributes:
        name (str):
        source (SourceEnum):
        description (Union[None, Unset, str]):
        huggingface_model_id (Union[None, Unset, str]):
        s3_address (Union[None, Unset, str]):
    """

    name: str
    source: SourceEnum
    description: Union[None, Unset, str] = UNSET
    huggingface_model_id: Union[None, Unset, str] = UNSET
    s3_address: Union[None, Unset, str] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        name = self.name

        source = self.source.value

        description: Union[None, Unset, str]
        if isinstance(self.description, Unset):
            description = UNSET
        else:
            description = self.description

        huggingface_model_id: Union[None, Unset, str]
        if isinstance(self.huggingface_model_id, Unset):
            huggingface_model_id = UNSET
        else:
            huggingface_model_id = self.huggingface_model_id

        s3_address: Union[None, Unset, str]
        if isinstance(self.s3_address, Unset):
            s3_address = UNSET
        else:
            s3_address = self.s3_address

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "source": source,
            }
        )
        if description is not UNSET:
            field_dict["description"] = description
        if huggingface_model_id is not UNSET:
            field_dict["huggingface_model_id"] = huggingface_model_id
        if s3_address is not UNSET:
            field_dict["s3_address"] = s3_address

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        name = d.pop("name")

        source = SourceEnum(d.pop("source"))

        def _parse_description(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        description = _parse_description(d.pop("description", UNSET))

        def _parse_huggingface_model_id(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        huggingface_model_id = _parse_huggingface_model_id(d.pop("huggingface_model_id", UNSET))

        def _parse_s3_address(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        s3_address = _parse_s3_address(d.pop("s3_address", UNSET))

        model_update = cls(
            name=name,
            source=source,
            description=description,
            huggingface_model_id=huggingface_model_id,
            s3_address=s3_address,
        )

        model_update.additional_properties = d
        return model_update

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
