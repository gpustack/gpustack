from typing import Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="ModelInstanceCreate")


@_attrs_define
class ModelInstanceCreate:
    """
    Attributes:
        model_id (int):
        node_id (Union[None, Unset, int]):
        pid (Union[None, Unset, int]):
        port (Union[None, Unset, int]):
        state (Union[None, Unset, str]):
    """

    model_id: int
    node_id: Union[None, Unset, int] = UNSET
    pid: Union[None, Unset, int] = UNSET
    port: Union[None, Unset, int] = UNSET
    state: Union[None, Unset, str] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        model_id = self.model_id

        node_id: Union[None, Unset, int]
        if isinstance(self.node_id, Unset):
            node_id = UNSET
        else:
            node_id = self.node_id

        pid: Union[None, Unset, int]
        if isinstance(self.pid, Unset):
            pid = UNSET
        else:
            pid = self.pid

        port: Union[None, Unset, int]
        if isinstance(self.port, Unset):
            port = UNSET
        else:
            port = self.port

        state: Union[None, Unset, str]
        if isinstance(self.state, Unset):
            state = UNSET
        else:
            state = self.state

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "model_id": model_id,
            }
        )
        if node_id is not UNSET:
            field_dict["node_id"] = node_id
        if pid is not UNSET:
            field_dict["pid"] = pid
        if port is not UNSET:
            field_dict["port"] = port
        if state is not UNSET:
            field_dict["state"] = state

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        model_id = d.pop("model_id")

        def _parse_node_id(data: object) -> Union[None, Unset, int]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, int], data)

        node_id = _parse_node_id(d.pop("node_id", UNSET))

        def _parse_pid(data: object) -> Union[None, Unset, int]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, int], data)

        pid = _parse_pid(d.pop("pid", UNSET))

        def _parse_port(data: object) -> Union[None, Unset, int]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, int], data)

        port = _parse_port(d.pop("port", UNSET))

        def _parse_state(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        state = _parse_state(d.pop("state", UNSET))

        model_instance_create = cls(
            model_id=model_id,
            node_id=node_id,
            pid=pid,
            port=port,
            state=state,
        )

        model_instance_create.additional_properties = d
        return model_instance_create

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
