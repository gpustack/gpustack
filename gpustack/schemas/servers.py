"""
Server相关的Schema定义

用于多Server环境下的Server实例管理和状态追踪。
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class ServerStatusEnum(str, Enum):
    """Server状态枚举"""
    ACTIVE = "active"  # 活跃
    INACTIVE = "inactive"  # 不活跃（心跳超时）
    MAINTENANCE = "maintenance"  # 维护中
    OFFLINE = "offline"  # 离线


class ServerInfo(BaseModel):
    """Server实例信息模型"""
    id: str = Field(..., description="Server唯一标识")
    name: str = Field(..., description="Server名称")
    address: str = Field(..., description="Server地址")
    api_port: int = Field(..., description="API端口")
    status: ServerStatusEnum = Field(
        default=ServerStatusEnum.ACTIVE, 
        description="Server状态"
    )
    last_heartbeat: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="最后心跳时间"
    )
    current_load: float = Field(default=0.0, description="当前负载分数")
    worker_count: int = Field(default=0, description="管理的Worker数量")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="创建时间"
    )
    
    class Config:
        from_attributes = True


class ServerCreate(BaseModel):
    """创建Server请求模型"""
    id: Optional[str] = Field(None, description="Server标识，不指定则自动生成")
    name: str = Field(..., description="Server名称")
    address: str = Field(..., description="Server地址")
    api_port: int = Field(..., description="API端口")


class ServerUpdate(BaseModel):
    """更新Server请求模型"""
    name: Optional[str] = Field(None, description="Server名称")
    status: Optional[ServerStatusEnum] = Field(None, description="Server状态")
    current_load: Optional[float] = Field(None, description="当前负载分数")
    worker_count: Optional[int] = Field(None, description="管理的Worker数量")


class ServerHeartbeat(BaseModel):
    """Server心跳请求模型"""
    server_id: str = Field(..., description="Server ID")
    name: str = Field(..., description="Server名称")
    address: str = Field(..., description="Server地址")
    api_port: int = Field(..., description="API端口")
    timestamp: datetime = Field(..., description="心跳时间戳")
    load_info: Optional[dict] = Field(None, description="负载信息")


class CoordinatorMessage(BaseModel):
    """协调消息模型"""
    type: str = Field(..., description="消息类型")
    source_server_id: str = Field(..., description="源Server ID")
    target_server_id: Optional[str] = Field(None, description="目标Server ID")
    payload: dict = Field(default_factory=dict, description="消息内容")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="消息时间戳"
    )


class DistributedLockInfo(BaseModel):
    """分布式锁信息模型"""
    lock_id: str = Field(..., description="锁标识")
    owner: str = Field(..., description="持有者")
    acquired_at: datetime = Field(..., description="获取时间")
    expires_at: datetime = Field(..., description="过期时间")
    timeout: int = Field(default=30, description="锁超时时间（秒）")


class ServerClusterInfo(BaseModel):
    """Server集群信息模型"""
    servers: List[ServerInfo] = Field(default_factory=list, description="Server列表")
    total_workers: int = Field(default=0, description="总Worker数量")
    total_load: float = Field(default=0.0, description="总负载")
    active_servers: int = Field(default=0, description="活跃Server数量")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="查询时间"
    )
