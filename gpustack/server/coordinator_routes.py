"""
协调服务API端点

提供多Server协调的REST API接口，包括：
- Server注册和心跳
- 分布式锁管理
- Server集群信息查询
- 协调消息处理
"""

from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel

from gpustack.schemas.servers import (
    ServerStatusEnum,
    ServerInfo,
    ServerCreate,
    ServerUpdate,
    ServerHeartbeat,
    CoordinatorMessage,
    DistributedLockInfo,
    ServerClusterInfo,
)
from gpustack.config.config import get_global_config

router = APIRouter(prefix="/coordinator", tags=["coordinator"])

# 内存中的协调服务状态（单Server模式）
_coordinator_state = {
    "servers": {},  # server_id -> ServerInfo
    "locks": {},    # lock_id -> DistributedLockInfo
}


class InMemoryCoordinatorService:
    """简单的内存协调服务（用于单Server或开发测试）"""
    
    def __init__(self):
        self._servers = {}
        self._locks = {}
    
    async def register_server(
        self,
        server_id: str,
        name: str,
        address: str,
        api_port: int,
    ) -> ServerInfo:
        """注册Server"""
        if server_id in self._servers:
            server = self._servers[server_id]
            server.name = name
            server.address = address
            server.api_port = api_port
            server.last_heartbeat = datetime.now(timezone.utc)
            server.status = ServerStatusEnum.ACTIVE
        else:
            server = ServerInfo(
                id=server_id,
                name=name,
                address=address,
                api_port=api_port,
                status=ServerStatusEnum.ACTIVE,
                last_heartbeat=datetime.now(timezone.utc),
            )
            self._servers[server_id] = server
        
        return server
    
    async def unregister_server(self, server_id: str):
        """注销Server"""
        if server_id in self._servers:
            del self._servers[server_id]
            # 清理该Server持有的锁
            expired_locks = [
                lock_id for lock_id, lock in self._locks.items()
                if lock.owner == server_id
            ]
            for lock_id in expired_locks:
                del self._locks[lock_id]
    
    async def heartbeat(
        self,
        server_id: str,
        name: str,
        address: str,
        api_port: int,
        timestamp: datetime,
        load_info: Optional[dict] = None,
    ) -> dict:
        """处理心跳"""
        now = datetime.now(timezone.utc)
        
        if server_id not in self._servers:
            # 自动注册
            await self.register_server(server_id, name, address, api_port)
        
        server = self._servers[server_id]
        server.last_heartbeat = now
        server.status = ServerStatusEnum.ACTIVE
        
        if load_info:
            if "load" in load_info:
                server.current_load = load_info["load"]
            if "worker_count" in load_info:
                server.worker_count = load_info["worker_count"]
        
        # 返回所有活跃Server的信息
        return {
            "servers": [
                {
                    "server_id": s.id,
                    "name": s.name,
                    "address": s.address,
                    "api_port": s.api_port,
                    "status": s.status.value,
                    "last_heartbeat": s.last_heartbeat.isoformat(),
                    "current_load": s.current_load,
                    "worker_count": s.worker_count,
                }
                for s in self._servers.values()
                if (now - s.last_heartbeat).total_seconds() < 60  # 60秒内有心跳
            ]
}
    
    async def acquire_lock(
        self,
        lock_id: str,
        owner: str,
        timeout: int = 30,
    ) -> bool:
        """获取锁"""
        now = datetime.now(timezone.utc)
        expires_at = now.fromtimestamp(now.timestamp() + timeout)
        
        if lock_id in self._locks:
            existing_lock = self._locks[lock_id]
            
            if now < existing_lock.expires_at:
                if existing_lock.owner == owner:
                    # 续期
                    existing_lock.expires_at = expires_at
                    return True
                return False
            else:
                del self._locks[lock_id]
        
        lock = DistributedLockInfo(
            lock_id=lock_id,
            owner=owner,
            acquired_at=now,
            expires_at=expires_at,
            timeout=timeout,
        )
        self._locks[lock_id] = lock
        return True
    
    async def release_lock(self, lock_id: str, owner: str) -> bool:
        """释放锁"""
        if lock_id not in self._locks:
            return False
        
        lock = self._locks[lock_id]
        if lock.owner != owner:
            return False
        
        del self._locks[lock_id]
        return True
    
    async def get_locks(self) -> List[DistributedLockInfo]:
        """获取所有锁"""
        return list(self._locks.values())
    
    async def get_servers(self) -> List[ServerInfo]:
        """获取所有Server"""
        now = datetime.now(timezone.utc)
        timeout_threshold = 60  # 60秒超时
        
        active_servers = []
        for server in self._servers.values():
            if (now - server.last_heartbeat).total_seconds() <= timeout_threshold:
                active_servers.append(server)
            else:
                server.status = ServerStatusEnum.INACTIVE
        
        return active_servers
    
    async def get_cluster_info(self) -> ServerClusterInfo:
        """获取集群信息"""
        servers = await self.get_servers()
        now = datetime.now(timezone.utc)
        
        total_workers = sum(s.worker_count for s in servers)
        total_load = sum(s.current_load for s in servers) / max(len(servers), 1)
        
        return ServerClusterInfo(
            servers=servers,
            total_workers=total_workers,
            total_load=total_load,
            active_servers=len(servers),
            timestamp=now,
        )


# 全局协调服务实例
_coordinator = InMemoryCoordinatorService()


@router.post("/servers/register")
async def register_server(request: ServerCreate):
    """注册新Server"""
    server = await _coordinator.register_server(
        server_id=request.id or f"server-{datetime.now(timezone.utc).timestamp()}",
        name=request.name,
        address=request.address,
        api_port=request.api_port,
    )
    return {"status": "registered", "server": server.model_dump()}


@router.post("/servers/unregister/{server_id}")
async def unregister_server(server_id: str):
    """注销Server"""
    await _coordinator.unregister_server(server_id)
    return {"status": "unregistered"}


@router.post("/heartbeat")
async def heartbeat(request: ServerHeartbeat):
    """处理Server心跳"""
    result = await _coordinator.heartbeat(
        server_id=request.server_id,
        name=request.name,
        address=request.address,
        api_port=request.api_port,
        timestamp=request.timestamp,
        load_info=request.load_info,
    )
    return result


@router.get("/servers")
async def get_servers():
    """获取所有Server"""
    servers = await _coordinator.get_servers()
    return {
        "servers": [s.model_dump() for s in servers],
        "count": len(servers),
    }


@router.get("/cluster")
async def get_cluster_info():
    """获取集群信息"""
    cluster_info = await _coordinator.get_cluster_info()
    return cluster_info.model_dump()


# ==================== 分布式锁API ====================

@router.post("/locks/acquire")
async def acquire_lock(lock_id: str, owner: str, timeout: int = 30):
    """获取分布式锁"""
    acquired = await _coordinator.acquire_lock(lock_id, owner, timeout)
    return {"acquired": acquired}


@router.post("/locks/release")
async def release_lock(lock_id: str, owner: str):
    """释放分布式锁"""
    released = await _coordinator.release_lock(lock_id, owner)
    return {"released": released}


@router.get("/locks")
async def get_locks():
    """获取所有锁"""
    locks = await _coordinator.get_locks()
    return {
        "locks": [lock.model_dump() for lock in locks],
        "count": len(locks),
    }


# ==================== 消息API ====================

@router.post("/message")
async def handle_message(message: CoordinatorMessage):
    """处理协调消息"""
    # 在实际实现中，这里会路由消息到对应的处理器
    logger.info(
        f"收到协调消息: type={message.type}, "
        f"source={message.source_server_id}, "
        f"target={message.target_server_id}"
    )
    
    # 返回确认
    return {
        "status": "received",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ==================== 健康检查 ====================

@router.get("/health")
async def health_check():
    """协调服务健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
