"""
多Server协调模块

为GPUStack Server添加多Server协调支持，包括：
- 协调服务集成
- 分布式调度
- Server心跳管理
- Worker联邦
"""

import asyncio
import logging
from typing import Optional, List
from datetime import datetime, timezone

from gpustack.config.config import Config
from gpustack.coordinator import (
    create_coordinator_service,
    CoordinatorService,
    ServerInfo,
)
from gpustack.scheduler.distributed_scheduler import DistributedScheduler
from gpustack.server.worker_syncer import WorkerSyncer

logger = logging.getLogger(__name__)


class MultiServerCoordinator:
    """
    多Server协调器
    
    管理单个Server实例与协调服务的集成，
    提供分布式调度和状态同步功能。
    """
    
    def __init__(self, config: Config):
        """
        初始化多Server协调器
        
        Args:
            config: GPUStack配置对象
        """
        self._config = config
        self._coordinator: Optional[CoordinatorService] = None
        self._scheduler: Optional[DistributedScheduler] = None
        self._server_id: Optional[str] = None
        self._running = False
    
    async def start(self):
        """启动协调器"""
        if self._running:
            logger.warning("协调器已经在运行")
            return
        
        self._running = True
        
        # 创建并启动协调服务
        self._server_id = self._config.server_id
        self._coordinator = create_coordinator_service(
            self._config,
            server_id=self._server_id
        )
        
        await self._coordinator.start()
        logger.info(
            f"协调服务已启动，Server ID: {self._coordinator.server_id}"
        )
        
        # 如果配置了分布式调度，创建分布式调度器
        if self._config.distributed_scheduling:
            self._scheduler = DistributedScheduler(self._config)
            self._scheduler.set_coordinator(
                self._coordinator,
                self._coordinator.server_id
            )
            logger.info("分布式调度器已配置")
    
    async def stop(self):
        """停止协调器"""
        if not self._running:
            return
        
        self._running = False
        
        # 停止协调服务
        if self._coordinator:
            await self._coordinator.stop()
            logger.info("协调服务已停止")
    
    @property
    def coordinator(self) -> Optional[CoordinatorService]:
        """获取协调服务实例"""
        return self._coordinator
    
    @property
    def scheduler(self) -> Optional[DistributedScheduler]:
        """获取分布式调度器实例"""
        return self._scheduler
    
    @property
    def server_id(self) -> Optional[str]:
        """获取Server ID"""
        return self._server_id
    
    @property
    def is_running(self) -> bool:
        """检查协调器是否在运行"""
        return self._running
    
    async def get_active_servers(self) -> List[ServerInfo]:
        """获取所有活跃的Server"""
        if self._coordinator:
            return await self._coordinator.get_active_servers()
        return []
    
    async def update_server_load(self, load: float, worker_count: int):
        """更新当前Server的负载信息"""
        if self._coordinator and self._server_id:
            await self._coordinator.update_server_status(
                self._server_id,
                load=load,
                worker_count=worker_count
            )
    
    async def broadcast_to_servers(self, message: dict):
        """广播消息到所有Server"""
        if self._coordinator:
            await self._coordinator.broadcast_message(message)


class ServerStateManager:
    """
    Server状态管理器
    
    管理多Server环境下的Server状态同步和健康检查。
    """
    
    def __init__(self, coordinator: CoordinatorService, server_id: str):
        """
        初始化状态管理器
        
        Args:
            coordinator: 协调服务实例
            server_id: 当前Server ID
        """
        self._coordinator = coordinator
        self._server_id = server_id
        self._check_interval = 30  # 检查间隔（秒）
        self._running = False
        self._task = None
    
    async def start(self):
        """启动状态管理器"""
        self._running = True
        self._task = asyncio.create_task(self._state_check_loop())
        logger.info("Server状态管理器已启动")
    
    async def stop(self):
        """停止状态管理器"""
        self._running = False
        if self._task:
            self._task.cancel()
        logger.info("Server状态管理器已停止")
    
    async def _state_check_loop(self):
        """状态检查循环"""
        while self._running:
            try:
                await self._sync_server_states()
                await asyncio.sleep(self._check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"状态检查失败: {e}")
                await asyncio.sleep(self._check_interval)
    
    async def _sync_server_states(self):
        """同步Server状态"""
        active_servers = await self._coordinator.get_active_servers()
        
        for server in active_servers:
            if server.id == self._server_id:
                continue
            
            # 可以在这里实现更复杂的状态同步逻辑
            # 例如：检查其他Server的健康状态、负载情况等
            logger.debug(
                f"Server {server.name} 状态: {server.status}, "
                f"负载: {server.current_load}, Worker数: {server.worker_count}"
            )


class WorkerFederationManager:
    """
    Worker联邦管理器
    
    管理跨Server的Worker状态同步和负载均衡。
    """
    
    def __init__(self, coordinator: CoordinatorService, server_id: str):
        """
        初始化Worker联邦管理器
        
        Args:
            coordinator: 协调服务实例
            server_id: 当前ServerID
        """
        self._coordinator = coordinator
        self._server_id = server_id
        self._sync_interval = 60  # 同步间隔（秒）
        self._running = False
        self._task = None
    
    async def start(self):
        """启动Worker联邦管理器"""
        self._running = True
        self._task = asyncio.create_task(self._sync_loop())
        logger.info("Worker联邦管理器已启动")
    
    async def stop(self):
        """停止Worker联邦管理器"""
        self._running = False
        if self._task:
            self._task.cancel()
        logger.info("Worker联邦管理器已停止")
    
    async def _sync_loop(self):
        """同步循环"""
        while self._running:
            try:
                await self._sync_worker_states()
                await asyncio.sleep(self._sync_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker状态同步失败: {e}")
                await asyncio.sleep(self._sync_interval)
    
    async def _sync_worker_states(self):
        """同步Worker状态"""
        # 获取全局Worker状态
        global_states = await self._coordinator.get_all_worker_states()
        
        # 可以在这里实现Worker状态合并和冲突解决逻辑
        logger.debug(
            f"同步了 {len(global_states)} 个Worker的全局状态"
        )
    
    async def register_worker_state(self, worker_id: str, state: dict):
        """注册Worker状态"""
        await self._coordinator.update_worker_state(worker_id, state)
    
    async def get_global_worker_states(self) -> dict:
        """获取全局Worker状态"""
        return await self._coordinator.get_all_worker_states()
