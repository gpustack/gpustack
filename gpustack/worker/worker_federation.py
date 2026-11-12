"""
Worker联邦模块

提供多Server环境下Worker的联邦管理功能，包括：
- 多Server注册
- 状态同步
- 故障转移
- 健康检查
"""

import asyncio
import logging
from typing import Optional, List, Dict
from datetime import datetime, timezone

from gpustack.config.config import Config
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.utils.network import is_url_reachable

logger = logging.getLogger(__name__)


class WorkerFederation:
    """
    Worker联邦管理器
    
    管理Worker在多Server环境下的注册和状态同步，
    支持故障转移和多Server容灾。
    """
    
    def __init__(self, config: Config):
        """
        初始化Worker联邦管理器
        
        Args:
            config: GPUStack配置对象
        """
        self._config = config
        self._server_urls: List[str] = []
        self._active_server: Optional[str] = None
        self._fallback_servers: List[str] = []
        self._registration_interval = 60  # 注册间隔（秒）
        self._health_check_interval = 30  # 健康检查间隔（秒）
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        # 初始化Server URL列表
        self._init_server_urls()
    
    def _init_server_urls(self):
        """初始化Server URL列表"""
        # 添加主Server URL
        if self._config.server_url:
            self._server_urls.append(self._config.server_url)
        
        # 添加配置文件中的其他Server URL
        if self._config.server_urls:
            for url in self._config.server_urls:
                if url not in self._server_urls:
                    self._server_urls.append(url)
        
        # 如果没有配置，使用本地Server
        if not self._server_urls:
            self._server_urls.append(f"http://127.0.0.1:{self._config.api_port}")
        
        logger.info(f"Worker联邦已配置，Server URLs: {self._server_urls}")
    
    async def start(self):
        """启动Worker联邦"""
        if self._running:
            logger.warning("Worker联邦已经在运行")
            return
        
        self._running = True
        
        # 选择最佳Server进行注册
        await self._select_best_server()
        
        # 启动注册任务
        self._tasks.append(asyncio.create_task(self._registration_loop()))
        
        # 启动健康检查任务
        self._tasks.append(asyncio.create_task(self._health_check_loop()))
        
        logger.info("Worker联邦已启动")
    
    async def stop(self):
        """停止Worker联邦"""
        if not self._running:
            return
        
        self._running = False
        
        # 取消所有任务
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        # 向所有Server注销
        await self._unregister_from_all()
        
        logger.info("Worker联邦已停止")
    
    async def _select_best_server(self):
        """选择最佳的Server进行注册"""
        if not self._server_urls:
            return
        
        if len(self._server_urls) == 1:
            self._active_server = self._server_urls[0]
            return
        
        # 测试所有Server的可达性
        server_latencies = []
        
        for server_url in self._server_urls:
            latency = await self._measure_server_latency(server_url)
            if latency is not None:
                server_latencies.append((server_url, latency))
        
        if server_latencies:
            # 选择延迟最低的Server
            server_latencies.sort(key=lambda x: x[1])
            self._active_server = server_latencies[0][0]
            self._fallback_servers = [s[0] for s in server_latencies[1:]]
        else:
            # 所有Server都不可达，使用第一个
            self._active_server = self._server_urls[0]
        
        logger.info(f"选择的活跃Server: {self._active_server}")
        if self._fallback_servers:
            logger.info(f"备用Server: {self._fallback_servers}")
    
    async def _measure_server_latency(self, server_url: str) -> Optional[float]:
        """测量Server延迟"""
        healthz_url = f"{server_url}/healthz"
        
        try:
            start = datetime.now()
            reachable = await is_url_reachable(healthz_url, timeout=5)
            if reachable:
                latency = (datetime.now() - start).total_seconds() * 1000
                return latency
        except Exception as e:
            logger.debug(f"无法连接到Server {server_url}: {e}")
        
        return None
    
    async def _registration_loop(self):
        """注册循环"""
        while self._running:
            try:
                # 向活跃Server注册
                if self._active_server:
                    success = await self._register_to_server(self._active_server)
                    if success:
                        logger.debug(f"成功注册到Server: {self._active_server}")
                    else:
                        # 注册失败，尝试故障转移
                        await self._handle_server_failure(self._active_server)
                
                await asyncio.sleep(self._registration_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"注册循环错误: {e}")
                await asyncio.sleep(self._registration_interval)
    
    async def _health_check_loop(self):
        """健康检查循环"""
        while self._running:
            try:
                await self._check_all_servers()
                await asyncio.sleep(self._health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"健康检查循环错误: {e}")
                await asyncio.sleep(self._health_check_interval)
    
    async def _check_all_servers(self):
        """检查所有Server的健康状态"""
        for server_url in self._server_urls:
            healthz_url = f"{server_url}/healthz"
            reachable = await is_url_reachable(healthz_url, timeout=5)
            
            if not reachable:
                logger.warning(f"Server {server_url} 不可达")
                if server_url == self._active_server:
                    await self._handle_server_failure(server_url)
    
    async def _handle_server_failure(self, failed_server: str):
        """处理Server故障"""
        logger.warning(f"处理Server故障: {failed_server}")
        
        # 移除失败的Server
        if failed_server in self._server_urls:
            self._server_urls.remove(failed_server)
        
        if failed_server == self._active_server:
            self._active_server = None
            
            # 选择新的活跃Server
            await self._select_best_server()
            
            if self._active_server:
                logger.info(f"故障转移到新Server: {self._active_server}")
    
    async def _register_to_server(self, server_url: str) -> bool:
        """向指定Server注册"""
        try:
            # 这里会调用Worker的实际注册逻辑
            # 由于Worker注册涉及认证等复杂逻辑，这里只提供框架
            logger.debug(f"尝试注册到Server: {server_url}")
            return True  # 实际实现需要调用Worker的注册API
        except Exception as e:
            logger.error(f"注册到Server {server_url} 失败: {e}")
            return False
    
    async def _unregister_from_all(self):
        """从所有Server注销"""
        for server_url in self._server_urls:
            try:
                logger.debug(f"从Server注销: {server_url}")
                # 实际实现需要调用Worker的注销API
            except Exception as e:
                logger.error(f"从Server {server_url} 注销失败: {e}")
    
    @property
    def active_server(self) -> Optional[str]:
        """获取当前活跃的Server"""
        return self._active_server
    
    @property
    def fallback_servers(self) -> List[str]:
        """获取备用Server列表"""
        return self._fallback_servers.copy()
    
    @property
    def all_servers(self) -> List[str]:
        """获取所有Server列表"""
        return self._server_urls.copy()


class MultiServerWorkerSelector:
    """
    多Server Worker选择器
    
    在多Server环境下提供Worker选择功能，
    支持跨Server的Worker查询和选择。
    """
    
    def __init__(self, config: Config):
        """
        初始化Worker选择器
        
        Args:
            config: GPUStack配置对象
        """
        self._config = config
        self._server_urls: List[str] = []
        self._init_server_urls()
    
    def _init_server_urls(self):
        """初始化Server URL列表"""
        if self._config.server_url:
            self._server_urls.append(self._config.server_url)
        
        if self._config.server_urls:
            self._server_urls.extend(self._config.server_urls)
        
        if not self._server_urls:
            self._server_urls.append(f"http://127.0.0.1:{self._config.api_port}")
    
    async def get_all_workers(self) -> List[Dict]:
        """从所有Server获取所有Worker"""
        all_workers = []
        
        for server_url in self._server_urls:
            try:
                workers = await self._fetch_workers_from_server(server_url)
                all_workers.extend(workers)
            except Exception as e:
                logger.error(f"从Server {server_url} 获取Worker失败: {e}")
        
        return all_workers
    
    async def get_available_workers(self) -> List[Dict]:
        """获取所有可用的Worker"""
        all_workers = await self.get_all_workers()
        
        available_workers = [
worker for worker in all_workers
            if worker.get("state") in [WorkerStateEnum.READY.value, WorkerStateEnum.PROVISIONING.value]
        ]
        
        return available_workers
    
    async def get_workers_by_capability(self, requirements: Dict) -> List[Dict]:
        """根据能力需求获取Worker"""
        workers = await self.get_available_workers()
        
        filtered_workers = []
        for worker in workers:
            if self._match_requirements(worker, requirements):
                filtered_workers.append(worker)
        
        return filtered_workers
    
    async def _fetch_workers_from_server(self, server_url: str) -> List[Dict]:
        """从指定Server获取Worker列表"""
        try:
            # 实际实现需要调用API
            # 这里返回空列表作为框架
            return []
        except Exception as e:
            logger.error(f"从Server {server_url} 获取Worker失败: {e}")
            return []
    
    def _match_requirements(self, worker: Dict, requirements: Dict) -> bool:
        """检查Worker是否匹配需求"""
        # GPU内存需求
        if "gpu_memory" in requirements:
            worker_gpu_memory = sum(
                gpu.get("memory", {}).get("total", 0)
                for gpu in worker.get("gpu_devices", [])
            )
            if worker_gpu_memory < requirements["gpu_memory"]:
                return False
        
        # GPU数量需求
        if "gpu_count" in requirements:
            gpu_count = len(worker.get("gpu_devices", []))
            if gpu_count < requirements["gpu_count"]:
                return False
        
        # 特定GPU类型需求
        if "gpu_type" in requirements:
            gpu_types = [
                gpu.get("type")
                for gpu in worker.get("gpu_devices", [])
            ]
            if requirements["gpu_type"] not in gpu_types:
                return False
        
        return True
