#!/usr/bin/env python3
"""
多Server功能验证脚本
用于验证多Server相关模块的导入和基本功能
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试所有新增模块的导入"""
    tests = []
    
    # 1. 测试Schema导入
    try:
        from gpustack.schemas.servers import (
            ServerStatusEnum,
            ServerInfo,
            ServerCreate,
            ServerHeartbeat,
            CoordinatorMessage,
            DistributedLockInfo,
            ServerClusterInfo,
        )
        tests.append(("✓ schemas.servers", True, ""))
    except Exception as e:
        tests.append(("✗ schemas.servers", False, str(e)))
    
    # 2. 测试协调服务导入
    try:
        from gpustack.coordinator import (
            CoordinatorService,
            LocalCoordinatorService,
            DistributedCoordinatorService,
            ServerInfo as CoordinatorServerInfo,
            DistributedLock as CoordinatorDistributedLock,
            SchedulingModeEnum,
        )
        tests.append(("✓ coordinator", True, ""))
    except Exception as e:
        tests.append(("✗ coordinator", False, str(e)))
    
    # 3. 测试分布式调度器导入
    try:
        from gpustack.scheduler.distributed_scheduler import (
            DistributedScheduler,
            find_candidate,
        )
        tests.append(("✓ distributed_scheduler", True, ""))
    except Exception as e:
        tests.append(("✗ distributed_scheduler", False, str(e)))
    
    # 4. 测试多Server协调模块导入
    try:
        from gpustack.server.multi_server import (
            MultiServerCoordinator,
            ServerStateManager,
            WorkerFederationManager,
        )
        tests.append(("✓ server.multi_server", True, ""))
    except Exception as e:
        tests.append(("✗ server.multi_server", False, str(e)))
    
    # 5. 测试Worker联邦导入
    try:
        from gpustack.worker.worker_federation import (
            WorkerFederation,
            MultiServerWorkerSelector,
        )
        tests.append(("✓ worker_federation", True, ""))
    except Exception as e:
        tests.append(("✗ worker_federation", False, str(e)))
    
    return tests


def test_config():
    """测试配置类"""
    try:
        from gpustack.config.config import Config
        
        # 测试配置项是否存在
        config_fields = Config.model_fields.keys()
        required_fields = [
            'server_id',
            'server_urls',
            'coordinator_url',
            'scheduling_mode',
            'heartbeat_interval',
            'server_timeout',
            'lock_timeout',
            'distributed_scheduling',
            'schedule_lock_timeout',
        ]
        
        missing = [f for f in required_fields if f not in config_fields]
        if missing:
            return False, f"缺少配置项: {missing}"
        return True, ""
    except Exception as e:
        return False, str(e)


def main():
    """主测试函数"""
    print("=" * 60)
    print("GPUStack 多Server功能验证")
    print("=" * 60)
    
    print("\n1. 模块导入测试")
    print("-" * 60)
    results = test_imports()
    for test, passed, error in results:
        status = "PASS" if passed else "FAIL"
        print(f"{test}: [{status}]")
        if not passed:
            print(f"   错误: {error}")
    
    print("\n2. 配置测试")
    print("-" * 60)
    passed, error = test_config()
    if passed:
        print("✓ Config类包含所有多Server配置项")
    else:
        print(f"✗ Config测试失败: {error}")
    
    # 统计结果
    total = len(results) + 1
    passed_count = sum(1 for r, _, _ in results if r.startswith("✓")) + (1 if passed else 0)
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed_count}/{total} 通过")
    
    if passed_count == total:
        print("✓ 所有测试通过！")
        return 0
    else:
        print("✗ 部分测试失败，请检查错误信息")
        return 1


if __name__ == "__main__":
    sys.exit(main())
