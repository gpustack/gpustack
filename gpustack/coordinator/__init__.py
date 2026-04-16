"""
分布式协调服务模块

提供多Server环境下的协调机制，包括：
- 服务注册与发现
- 分布式锁管理
- Server心跳检测
- Worker状态同步
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import aiohttp

from gpustack.config.config import Config
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.schemas.servers import ServerStatusEnum

logger = logging.getLogger(__name__)


class SchedulingModeEnum(str, Enum):
    """调度模式枚举"""
    LOCAL = "local"  # 仅本地调度
    DISTRIBUTED = "distributed"  # 分布式调度
    AUTO = "auto"  # 自动选择（根据负载）


@dataclass
class ServerInfo:
    """Server实例信息"""
    id: str
    name: str
    address: str
    api_port: int
    status: ServerStatusEnum
    last_heartbeat: datetime
    current_load: float = 0.0
    worker_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class DistributedLock:
    """分布式锁信息"""
    lock_id: str
    owner: str
    acquired_at: datetime
    expires_at: datetime
    timeout: int = 30  # 默认30秒超时


class CoordinatorService:
    """
    分布式协调服务
    
    负责管理多Server环境下的协调工作，包括：
    - Server注册与注销
    - 分布式锁管理
    - 心跳检测
    - 负载信息收集
    """
    
    def __init__(self, config: Config, server_id: Optional[str] = None):
        """
        初始化协调服务
        
        Args:
            config: GPUStack配置对象
            server_id: 当前Server的唯一标识，如果为None则自动生成
        """
        self._config = config
        self._server_id = server_id or str(uuid.uuid4())
        self._server_name = f"gpustack-{self._server_id[:8]}"
        
        # Server注册表
        self._servers: Dict[str, ServerInfo] = {}
        
        # 分布式锁表
        self._locks: Dict[str, DistributedLock] = {}
        
        # Worker状态缓存（用于跨Server同步）
        self._worker_states: Dict[str, Dict] = {}
        
        # 心跳间隔（秒）
        self._heartbeat_interval = config.heartbeat_interval or 15
        
        # Server超时时间（秒）
        self._server_timeout = config.server_timeout or 60
        
        # 锁超时时间（秒）
        self._lock_timeout = 30
        
        # 自身注册信息
        self._self_info: Optional[ServerInfo] = None
        
        # HTTP会话
        self._session: Optional[aiohttp.ClientSession] = None
        
        # 运行状态
        self._running = False
        
        # 异步任务
        self._tasks: List[asyncio.Task] = []
    
    @property
    def server_id(self) -> str:
        """获取当前Server的ID"""
        return self._server_id
    
    @property
    def server_name(self) -> str:
        """获取当前Server的名称"""
        return self._server_name
    
    async def start(self):
        """启动协调服务"""
        if self._running:
            logger.warning("协调服务已经在运行中")
            return
        
        self._running = True
        self._session = aiohttp.ClientSession()
        
        # 注册当前Server
        await self._register_self()
        
        # 启动心跳任务
        self._tasks.append(asyncio.create_task(self._heartbeat_loop()))
        
        # 启动Server超时检测任务
        self._tasks.append(asyncio.create_task(self._server_timeout_check_loop()))
        
        # 启动锁过期检测任务
        self._tasks.append(asyncio.create_task(self._lock_expiry_check_loop()))
        
        # 启动集群形成任务
        self._tasks.append(asyncio.create_task(self._cluster_formation_loop()))
        
        logger.info(f"协调服务已启动，Server ID: {self._server_id}")
    
    async def stop(self):
        """停止协调服务"""
        if not self._running:
            return
        
        self._running = False
        
        # 取消所有任务
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        # 注销当前Server
        await self._unregister_self()
        
        # 关闭HTTP会话
        if self._session:
            await self._session.close()
            self._session = None
        
        logger.info("协调服务已停止")
    
    async def _register_self(self):
        """注册当前Server到协调服务"""
        address = self._config.advertise_address or "127.0.0.1"
        
        self._self_info = ServerInfo(
            id=self._server_id,
            name=self._server_name,
            address=address,
            api_port=self._config.api_port or 30080,
            status=ServerStatusEnum.ACTIVE,
            last_heartbeat=datetime.now(timezone.utc),
        )
        
        # 在本地注册表中注册
        self._servers[self._server_id] = self._self_info
        
        logger.info(f"Server {self._server_name} ({self._server_id}) 已注册")
    
    async def _unregister_self(self):
        """注销当前Server"""
        if self._server_id in self._servers:
            del self._servers[self._server_id]
            logger.info(f"Server {self._server_name} ({self._server_id}) 已注销")
    
    async def _heartbeat_loop(self):
        """心跳发送循环"""
        while self._running:
            try:
                await self._send_heartbeat()
                await asyncio.sleep(self._heartbeat_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"发送心跳失败: {e}", exc_info=True)
                logger.debug(f"心跳间隔: {self._heartbeat_interval}秒")
                await asyncio.sleep(self._heartbeat_interval)
    
    async def _send_heartbeat(self):
        """发送心跳"""
        # 如果配置了协调服务URL，则发送到协调服务
        if self._config.coordinator_url:
            try:
                async with self._session.post(
                    f"{self._config.coordinator_url}/api/v1/heartbeat",
                    json={
                        "server_id": self._server_id,
                        "name": self._server_name,
                        "address": self._config.advertise_address,
                        "api_port": self._config.api_port,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        # 更新其他Server信息
                        self._update_servers_from_response(data.get("servers", []))
            except Exception as e:
                logger.warning(f"无法连接到协调服务: {e}")
        
        # 更新本地心跳时间
        if self._server_id in self._servers:
            self._servers[self._server_id].last_heartbeat = datetime.now(timezone.utc)
    
    def _update_servers_from_response(self, servers_data: List[Dict]):
        """从协调服务响应更新Server列表"""
        for server_data in servers_data:
            server_id = server_data.get("server_id")
            if server_id and server_id != self._server_id:
                if server_id not in self._servers:
                    self._servers[server_id] = ServerInfo(
                        id=server_id,
                        name=server_data.get("name", f"server-{server_id[:8]}"),
                        address=server_data.get("address", ""),
                        api_port=server_data.get("api_port", 30080),
                        status=ServerStatusEnum.ACTIVE,
                        last_heartbeat=datetime.fromisoformat(
                            server_data.get("last_heartbeat", datetime.now(timezone.utc).isoformat())
                        ),
                    )
    
    async def _server_timeout_check_loop(self):
        """Server超时检测循环"""
        while self._running:
            try:
                await self._check_server_timeouts()
                await asyncio.sleep(self._heartbeat_interval * 2)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"检测Server超时失败: {e}", exc_info=True)
                logger.debug(f"超时检测间隔: {self._heartbeat_interval * 2}秒")
                await asyncio.sleep(self._heartbeat_interval * 2)
    
    async def _check_server_timeouts(self):
        """检测超时的Server"""
        now = datetime.now(timezone.utc)
        timeout_threshold = timedelta(seconds=self._server_timeout)
        
        for server_id, server in list(self._servers.items()):
            if server_id == self._server_id:
                continue
            
            if now - server.last_heartbeat > timeout_threshold:
                # Server超时，标记为不活跃
                server.status = ServerStatusEnum.INACTIVE
                logger.warning(
                    f"Server {server.name} ({server_id}) 已超时，心跳时间: "
                    f"{server.last_heartbeat.isoformat()}"
                )
    
    async def _lock_expiry_check_loop(self):
        """锁过期检测循环"""
        while self._running:
            try:
                await self._check_lock_expiry()
                await asyncio.sleep(10)  # 每10秒检查一次
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"检测锁过期失败: {e}", exc_info=True)
                await asyncio.sleep(10)
    
    async def _check_lock_expiry(self):
        """检测过期的锁并清理"""
        now = datetime.now(timezone.utc)
        expired_locks = []
        
        for lock_id, lock in self._locks.items():
            if now >= lock.expires_at:
                expired_locks.append(lock_id)
        
        for lock_id in expired_locks:
            lock = self._locks.pop(lock_id)
            logger.info(f"锁 {lock_id} 已过期，由 {lock.owner} 持有")
    
    async def _cluster_formation_loop(self):
        """集群形成循环
        
        Cluster formation loop
        
        新增：定期尝试发现其他 Server，解决 Server 启动顺序问题
        Added: Periodically try to discover other servers to handle server startup order issues
        """
        while self._running:
            try:
                await self._ensure_cluster_formation()
                await asyncio.sleep(30)  # 每30秒尝试一次
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"集群形成循环失败: {e}", exc_info=True)
                logger.debug(f"集群形成间隔: 30秒")
                await asyncio.sleep(30)
    
    async def _ensure_cluster_formation(self):
        """确保集群形成，处理 Server 启动顺序问题
        
        Ensure cluster formation, handle server startup order issues
        
        新增：定期尝试发现其他 Server，确保所有 Server 都能相互发现
        Added: Periodically try to discover other servers to ensure all servers can find each other
        """
        # 定期尝试发现其他 Server
        # Periodically try to discover other servers
        if hasattr(self._config, 'server_urls') and self._config.server_urls:
            for server_url in self._config.server_urls:
                if server_url != self._get_self_address():
                    await self._discover_server(server_url)
    
    def _get_self_address(self):
        """获取当前Server的地址"""
        address = getattr(self._config, 'advertise_address', '127.0.0.1')
        api_port = getattr(self._config, 'api_port', 30080)
        return f"http://{address}:{api_port}"
    
    async def _discover_server(self, server_url):
        """发现其他 Server"""
        try:
            async with self._session.get(
                f"{server_url}/api/v1/coordinator/servers",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    for server_data in data.get("servers", []):
                        server_id = server_data.get("id")
                        if server_id != self._server_id:
                            # 注册发现的 Server
                            await self.register_server(
                                server_id=server_id,
                                name=server_data.get("name"),
                                address=server_data.get("address"),
                                api_port=server_data.get("api_port"),
                            )
        except Exception as e:
            logger.debug(f"发现 Server {server_url} 失败: {e}")
    
    # ==================== Server管理API ====================
    
    async def register_server(
        self,
        server_id: str,
        name: str,
        address: str,
        api_port: int,
    ) -> ServerInfo:
        """
        注册新的Server实例
        
        Args:
            server_id: Server唯一标识
            name: Server名称
            address: Server地址
            api_port: API端口
        
        Returns:
            ServerInfo: 注册的Server信息
        """
        now = datetime.now(timezone.utc)
        
        server = ServerInfo(
            id=server_id,
            name=name,
            address=address,
            api_port=api_port,
            status=ServerStatusEnum.ACTIVE,
            last_heartbeat=now,
        )
        
        self._servers[server_id] = server
        
        logger.info(f"Server {name} ({server_id}) 已注册到协调服务")
        return server
    
    async def unregister_server(self, server_id: str):
        """
        注销Server实例
        
        Args:
            server_id: 要注销的Server ID
        """
        if server_id in self._servers:
            server = self._servers.pop(server_id)
            logger.info(f"Server {server.name} ({server_id}) 已从协调服务注销")
            
            # 清理该Server持有的所有锁
            expired_locks = [
                lock_id for lock_id, lock in self._locks.items()
                if lock.owner == server_id
            ]
            for lock_id in expired_locks:
                del self._locks[lock_id]
    
    async def update_server_status(
        self,
        server_id: str,
        status: Optional[ServerStatusEnum] = None,
        load: Optional[float] = None,
        worker_count: Optional[int] = None,
    ) -> Optional[ServerInfo]:
        """
        更新Server状态
        
        Args:
            server_id: Server ID
            status: 新状态
            load: 负载信息
            worker_count: Worker数量
        
        Returns:
            更新后的Server信息，如果不存在则返回None
        """
        if server_id not in self._servers:
            return None
        
        server = self._servers[server_id]
        now = datetime.now(timezone.utc)
        
        if status is not None:
            server.status = status
        
        if load is not None:
            server.current_load = load
        
        if worker_count is not None:
            server.worker_count = worker_count
        
        server.last_heartbeat = now
        
        return server
    
    async def get_active_servers(self) -> List[ServerInfo]:
        """
        获取所有活跃的Server列表
        
        Returns:
            活跃Server列表
        """
        now = datetime.now(timezone.utc)
        timeout_threshold = timedelta(seconds=self._server_timeout)
        
        active_servers = []
        for server in self._servers.values():
            if server.status == ServerStatusEnum.ACTIVE:
                if now - server.last_heartbeat <= timeout_threshold:
                    active_servers.append(server)
                else:
                    # 标记为超时
                    server.status = ServerStatusEnum.INACTIVE
        
        return active_servers
    
    async def get_server_info(self, server_id: str) -> Optional[ServerInfo]:
        """
        获取指定Server的信息
        
        Args:
            server_id: Server ID
        
        Returns:
            Server信息，如果不存在则返回None
        """
        return self._servers.get(server_id)
    
    async def get_all_servers(self) -> List[ServerInfo]:
        """
        获取所有已注册的Server列表
        
        Returns:
            所有Server列表（包括超时的）
        """
        return list(self._servers.values())
    
    # ==================== 分布式锁API ====================
    
    async def acquire_lock(
        self,
        lock_id: str,
        owner: str,
        timeout: Optional[int] = None,
    ) -> bool:
        """
        获取分布式锁
        
        Args:
            lock_id: 锁的唯一标识
            owner: 锁的持有者（通常是Server ID）
            timeout: 锁的超时时间（秒），默认30秒
        
        Returns:
            是否成功获取锁
        """
        timeout = timeout or self._lock_timeout
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=timeout)
        
        if lock_id in self._locks:
            existing_lock = self._locks[lock_id]
            
            # 检查锁是否过期
            if now < existing_lock.expires_at:
                # 锁未过期，检查是否是同一个持有者
                if existing_lock.owner == owner:
                    # 续期锁
                    existing_lock.expires_at = expires_at
                    logger.debug(f"锁 {lock_id} 已续期")
                    return True
                else:
                    logger.debug(f"锁 {lock_id} 已被 {existing_lock.owner} 持有")
                    return False
            else:
                # 锁已过期，清理旧锁
                del self._locks[lock_id]
        
        # 创建新锁
        new_lock = DistributedLock(
            lock_id=lock_id,
            owner=owner,
            acquired_at=now,
            expires_at=expires_at,
            timeout=timeout,
        )
        self._locks[lock_id] = new_lock
        
        logger.debug(f"锁 {lock_id} 已被 {owner} 获取")
        return True
    
    async def release_lock(self, lock_id: str, owner: str) -> bool:
        """
        释放分布式锁
        
        Args:
            lock_id: 锁的标识
            owner: 锁的持有者
        
        Returns:
            是否成功释放锁
        """
        if lock_id not in self._locks:
            logger.warning(f"锁 {lock_id} 不存在")
            return False
        
        lock = self._locks[lock_id]
        
        if lock.owner != owner:
            logger.warning(
                f"无法释放锁 {lock_id}：持有者是 {lock.owner}，请求者是 {owner}"
            )
            return False
        
        del self._locks[lock_id]
        logger.debug(f"锁 {lock_id} 已被 {owner} 释放")
        return True
    
    async def extend_lock(self, lock_id: str, owner: str, additional_time: int) -> bool:
        """
        延长锁的持有时间
        
        Args:
            lock_id: 锁的标识
            owner: 锁的持有者
            additional_time: 延长时间（秒）
        
        Returns:
            是否成功延长
        """
        if lock_id not in self._locks:
            return False
        
        lock = self._locks[lock_id]
        
        if lock.owner != owner:
            return False
        
        lock.expires_at = datetime.now(timezone.utc) + timedelta(seconds=additional_time)
        return True
    
    async def get_lock_info(self, lock_id: str) -> Optional[DistributedLock]:
        """
        获取锁的信息
        
        Args:
            lock_id: 锁的标识
        
        Returns:
            锁信息，如果不存在则返回None
        """
        return self._locks.get(lock_id)
    
    async def get_locks_by_owner(self, owner: str) -> List[DistributedLock]:
        """
        获取指定持有者的所有锁
        
        Args:
            owner: 持有者标识
        
        Returns:
            锁列表
        """
        return [lock for lock in self._locks.values() if lock.owner == owner]
    
    # ==================== Worker状态同步API ====================
    
    async def update_worker_state(self, worker_id: str, state: Dict):
        """
        更新Worker状态
        
        Args:
            worker_id: Worker ID
            state: 状态信息字典
        """
        self._worker_states[worker_id] = {
            "state": state,
            "updated_at": datetime.now(timezone.utc),
        }
    
    async def get_worker_state(self, worker_id: str) -> Optional[Dict]:
        """
        获取Worker状态
        
        Args:
            worker_id: Worker ID
        
        Returns:
            Worker状态信息，如果不存在则返回None
        """
        return self._worker_states.get(worker_id)
    
    async def get_all_worker_states(self) -> Dict[str, Dict]:
        """
        获取所有Worker状态
        
        Returns:
            Worker状态字典
        """
        return self._worker_states.copy()
    
    async def remove_worker_state(self, worker_id: str):
        """
        移除Worker状态
        
        Args:
            worker_id: Worker ID
        """
        if worker_id in self._worker_states:
            del self._worker_states[worker_id]
    
    # ==================== 负载均衡API ====================
    
    async def calculate_server_load(self, server_id: str) -> float:
        """
        计算指定Server的负载分数
        
        Args:
            server_id: Server ID
        
        Returns:
            负载分数（0.0-1.0之间）
        """
        if server_id not in self._servers:
            return 1.0  # 如果Server不存在，返回最高负载
        
        server = self._servers[server_id]
        
        # 负载计算公式：基于Worker数量和活跃实例数
        worker_factor = min(server.worker_count / 10.0, 1.0) if server.worker_count else 0.0
        load_factor = server.current_load
        
        return (worker_factor * 0.4 + load_factor * 0.6)
    
    async def select_best_server(
        self,
        prefer_local: bool = True,
    ) -> Optional[ServerInfo]:
        """
        选择最优的Server进行调度
        
        Args:
            prefer_local: 是否优先选择本地Server
        
        Returns:
            最优Server信息，如果没有活跃Server则返回None
        """
        active_servers = await self.get_active_servers()
        
        if not active_servers:
            return None
        
        # 如果只有一个活跃Server，直接返回
        if len(active_servers) == 1:
            return active_servers[0]
        
        # 优先本地Server
        if prefer_local and self._server_id in [s.id for s in active_servers]:
            return self._servers[self._server_id]
        
        # 按负载排序，选择负载最低的
        server_loads = await asyncio.gather(
            *[(self.calculate_server_load(s.id), s) for s in active_servers]
        )
        
        best_server = min(server_loads, key=lambda x: x[0])
        return best_server[1]
    
    async def broadcast_message(self, message: Dict, exclude_self: bool = True):
        """
        广播消息到所有Server
        
        Args:
            message: 消息内容
            exclude_self: 是否排除自己
        """
        active_servers = await self.get_active_servers()
        
        for server in active_servers:
            if exclude_self and server.id == self._server_id:
                continue
            
            try:
                async with self._session.post(
                    f"http://{server.address}:{server.api_port}/api/v1/coordinator/message",
                    json=message,
                    timeout=aiohttp.ClientTimeout(total=5),
                ):
                    pass  # 忽略响应
            except Exception as e:
                logger.warning(f"无法广播消息到 Server {server.name}: {e}")


class LocalCoordinatorService(CoordinatorService):
    """
    本地协调服务实现
    
    用于单Server模式或没有外部协调服务的情况，
    所有协调逻辑在本地内存中执行。
    """
    
    def __init__(self, config: Config, server_id: Optional[str] = None):
        """
        初始化本地协调服务
        
        Args:
            config: GPUStack配置对象
            server_id: 当前Server的唯一标识
        """
        super().__init__(config, server_id)
        
        # 本地模式，不需要外部协调服务
        self._config.coordinator_url = None
        
        logger.info("使用本地协调服务模式")


class DistributedCoordinatorService(CoordinatorService):
    """
    分布式协调服务实现
    
    使用外部协调服务（如etcd、Consul、ZooKeeper）
    来实现真正的分布式协调。
    """
    
    def __init__(self, config: Config, server_id: Optional[str] = None):
        """
        初始化分布式协调服务
        
        Args:
            config: GPUStack配置对象
            server_id: 当前Server的唯一标识
        """
        super().__init__(config, server_id)
        
        if not config.coordinator_url:
            raise ValueError("分布式协调服务需要配置 coordinator_url")
        
        logger.info(f"使用分布式协调服务，URL: {config.coordinator_url}")


def create_coordinator_service(
    config: Config,
    server_id: Optional[str] = None,
) -> CoordinatorService:
    """
    创建协调服务实例
    
    Args:
        config: GPUStack配置对象
        server_id: Server唯一标识
    
    Returns:
        CoordinatorService实例
    """
    if config.coordinator_url:
        return DistributedCoordinatorService(config, server_id)
    else:
        return LocalCoordinatorService(config, server_id)
