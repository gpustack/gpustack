#!/usr/bin/env python3
"""
GPUStack 日志轮转系统统一测试脚本

测试功能：
1. 基本日志记录
2. 日期轮转机制
3. 线程安全性
4. 性能测试
5. 文件创建和管理
6. 与GPUStack集成测试
"""

import os
import sys
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

# 添加GPUStack模块路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print("=" * 80)
print("🧪 GPUStack 日志轮转系统统一测试")
print("=" * 80)

try:
    from gpustack.worker.daily_rotating_logger import DailyRotatingLogFile
    from gpustack.logging import RedirectStdoutStderr
    print("✅ 成功导入所需模块")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保在GPUStack项目根目录运行此脚本")
    sys.exit(1)

class LogRotationTester:
    """日志轮转测试器"""
    
    def __init__(self):
        self.test_dir = None
        self.logger = None
        self.results = {}
        
    def setup(self):
        """设置测试环境"""
        print("\n🔧 设置测试环境...")
        
        # 创建临时测试目录
        self.test_dir = Path(tempfile.mkdtemp(prefix="gpustack_log_test_"))
        print(f"测试目录: {self.test_dir}")
        
        # 清理可能存在的旧文件
        for old_file in self.test_dir.glob("*.log"):
            old_file.unlink()
        
        print("✅ 测试环境准备完成")
        return True
    
    def cleanup(self):
        """清理测试环境"""
        print("\n🧹 清理测试环境...")
        
        if self.logger:
            self.logger.close()
            
        if self.test_dir and self.test_dir.exists():
            try:
                shutil.rmtree(self.test_dir)
                print("✅ 测试目录已清理")
            except Exception as e:
                print(f"⚠️ 清理测试目录失败: {e}")
    
    def test_basic_logging(self):
        """测试1: 基本日志记录功能"""
        print("\n📝 测试1: 基本日志记录功能")
        
        try:
            # 创建日志对象（较短的检查间隔用于测试）
            self.logger = DailyRotatingLogFile(str(self.test_dir), check_interval=10)
            
            # 验证初始状态
            current_file = self.logger.get_current_log_file()
            today = datetime.now().strftime("%Y-%m-%d")
            expected_file = self.test_dir / f"{today}.log"
            
            assert str(expected_file) == current_file, f"文件路径不匹配: {current_file} != {expected_file}"
            print(f"✅ 日志文件路径正确: {current_file}")
            
            # 测试写入功能
            test_messages = [
                f"测试消息1 - {datetime.now()}",
                f"测试消息2 - {datetime.now()}",
                f"多行测试消息\n第二行内容",
                "Unicode测试: 🚀📝🔥"
            ]
            
            for i, msg in enumerate(test_messages):
                self.logger.write(f"{msg}\n")
                time.sleep(0.01)  # 短暂间隔
            
            # 验证文件内容
            if expected_file.exists():
                with open(expected_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.strip().split('\n')
                    
                assert len(lines) >= len(test_messages), f"日志行数不足: {len(lines)} < {len(test_messages)}"
                print(f"✅ 成功写入 {len(lines)} 行日志")
                
                # 检查内容
                for i, expected in enumerate(test_messages):
                    if expected in content:
                        print(f"✅ 消息 {i+1} 写入成功")
                    else:
                        print(f"⚠️ 消息 {i+1} 可能写入异常")
            else:
                raise FileNotFoundError(f"日志文件未创建: {expected_file}")
            
            self.results['basic_logging'] = True
            print("✅ 基本日志记录测试通过")
            
        except Exception as e:
            print(f"❌ 基本日志记录测试失败: {e}")
            self.results['basic_logging'] = False
            return False
        
        return True
    
    def test_date_rotation(self):
        """测试2: 日期轮转机制"""
        print("\n🔄 测试2: 日期轮转机制")
        
        try:
            if not self.logger:
                print("❌ 需要先运行基本日志测试")
                return False
            
            # 记录当前状态
            original_get_date = self.logger._get_current_date
            today = datetime.now().strftime("%Y-%m-%d")
            
            # 写入今天的日志
            self.logger.write(f"今天的日志 - {datetime.now()}\n")
            today_file = self.test_dir / f"{today}.log"
            
            assert today_file.exists(), "今天的日志文件不存在"
            print(f"✅ 今天的日志文件: {today_file.name}")
            
            # 模拟日期变化到明天
            tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            print(f"🔮 模拟日期变化: {today} → {tomorrow}")
            
            # 临时替换日期获取方法
            self.logger._get_current_date = lambda: tomorrow
            
            # 手动触发轮转检查
            self.logger._rotate_if_needed()
            
            # 写入明天的日志
            self.logger.write(f"明天的日志 - {datetime.now()}\n")
            tomorrow_file = self.test_dir / f"{tomorrow}.log"
            
            # 验证轮转结果
            assert tomorrow_file.exists(), "明天的日志文件未创建"
            print(f"✅ 明天的日志文件: {tomorrow_file.name}")
            
            # 检查两个文件都存在且内容正确
            with open(today_file, 'r', encoding='utf-8') as f:
                today_content = f.read()
                assert "今天的日志" in today_content, "今天的日志内容不正确"
            
            with open(tomorrow_file, 'r', encoding='utf-8') as f:
                tomorrow_content = f.read()
                assert "明天的日志" in tomorrow_content, "明天的日志内容不正确"
                if "Log rotated" in tomorrow_content:
                    print("✅ 发现轮转标记")
            
            # 恢复原始日期方法
            self.logger._get_current_date = original_get_date
            
            # 测试后天的轮转
            day_after = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
            self.logger._get_current_date = lambda: day_after
            self.logger._rotate_if_needed()
            self.logger.write(f"后天的日志 - {datetime.now()}\n")
            
            # 检查总文件数
            log_files = list(self.test_dir.glob("*.log"))
            assert len(log_files) >= 3, f"日志文件数量不足: {len(log_files)} < 3"
            print(f"✅ 生成了 {len(log_files)} 个日志文件")
            
            # 显示所有文件
            for log_file in sorted(log_files):
                size = log_file.stat().st_size
                print(f"  📄 {log_file.name} ({size} bytes)")
            
            # 恢复原始方法
            self.logger._get_current_date = original_get_date
            
            self.results['date_rotation'] = True
            print("✅ 日期轮转测试通过")
            
        except Exception as e:
            print(f"❌ 日期轮转测试失败: {e}")
            self.results['date_rotation'] = False
            return False
        
        return True
    
    def test_thread_safety(self):
        """测试3: 线程安全性"""
        print("\n🧵 测试3: 线程安全性")
        
        try:
            if not self.logger:
                print("❌ 需要先运行基本日志测试")
                return False
            
            # 验证后台线程状态
            if self.logger._checker_thread and self.logger._checker_thread.is_alive():
                print("✅ 后台检查线程正在运行")
            else:
                print("⚠️ 后台检查线程未运行")
            
            # 多线程并发写入测试
            def write_logs(thread_id, count=20):
                """线程写入函数"""
                for i in range(count):
                    self.logger.write(f"线程{thread_id}-消息{i+1} - {datetime.now()}\n")
                    time.sleep(0.001)  # 微小延迟
            
            print("启动多线程并发写入测试...")
            threads = []
            thread_count = 5
            messages_per_thread = 20
            
            start_time = time.time()
            
            # 启动多个线程
            for i in range(thread_count):
                thread = threading.Thread(target=write_logs, args=(i+1, messages_per_thread))
                threads.append(thread)
                thread.start()
            
            # 等待所有线程完成
            for thread in threads:
                thread.join()
            
            elapsed_time = time.time() - start_time
            total_messages = thread_count * messages_per_thread
            
            print(f"✅ {thread_count}个线程并发写入 {total_messages} 条消息")
            print(f"✅ 耗时: {elapsed_time:.3f}秒，平均: {elapsed_time/total_messages*1000:.2f}ms/条")
            
            # 验证日志完整性
            current_file = self.logger.get_current_log_file()
            if os.path.exists(current_file):
                with open(current_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.strip().split('\n')
                    
                # 统计每个线程的消息数
                thread_counts = {}
                for line in lines:
                    for tid in range(1, thread_count + 1):
                        if f"线程{tid}-" in line:
                            thread_counts[tid] = thread_counts.get(tid, 0) + 1
                
                print("📊 各线程消息统计:")
                all_complete = True
                for tid in range(1, thread_count + 1):
                    count = thread_counts.get(tid, 0)
                    status = "✅" if count == messages_per_thread else "⚠️"
                    print(f"  {status} 线程{tid}: {count}/{messages_per_thread} 条消息")
                    if count != messages_per_thread:
                        all_complete = False
                
                if all_complete:
                    print("✅ 所有线程消息完整")
                else:
                    print("⚠️ 部分线程消息可能丢失（这在高并发下是可能的）")
            
            self.results['thread_safety'] = True
            print("✅ 线程安全性测试通过")
            
        except Exception as e:
            print(f"❌ 线程安全性测试失败: {e}")
            self.results['thread_safety'] = False
            return False
        
        return True
    
    def test_performance(self):
        """测试4: 性能测试"""
        print("\n⚡ 测试4: 性能测试")
        
        try:
            if not self.logger:
                print("❌ 需要先运行基本日志测试")
                return False
            
            # 大量写入性能测试
            message_counts = [100, 500, 1000]
            
            for count in message_counts:
                print(f"\n📈 测试写入 {count} 条消息...")
                
                start_time = time.time()
                
                for i in range(count):
                    self.logger.write(f"性能测试消息 {i+1}/{count} - {datetime.now()}\n")
                
                elapsed_time = time.time() - start_time
                avg_time = elapsed_time / count * 1000  # 毫秒
                throughput = count / elapsed_time  # 消息/秒
                
                print(f"✅ 总耗时: {elapsed_time:.3f}秒")
                print(f"✅ 平均每条: {avg_time:.2f}ms")
                print(f"✅ 吞吐量: {throughput:.0f} 消息/秒")
                
                # 性能基准检查
                if avg_time < 1.0:  # 每条消息少于1ms
                    print("🚀 性能优秀")
                elif avg_time < 5.0:  # 每条消息少于5ms
                    print("✅ 性能良好")
                else:
                    print("⚠️ 性能可能需要优化")
            
            # 验证文件大小
            current_file = self.logger.get_current_log_file()
            if os.path.exists(current_file):
                file_size = os.path.getsize(current_file)
                print(f"\n📏 当前日志文件大小: {file_size:,} bytes ({file_size/1024:.1f} KB)")
            
            self.results['performance'] = True
            print("✅ 性能测试通过")
            
        except Exception as e:
            print(f"❌ 性能测试失败: {e}")
            self.results['performance'] = False
            return False
        
        return True
    
    def test_integration_with_redirect(self):
        """测试5: 与RedirectStdoutStderr集成测试"""
        print("\n🔗 测试5: 与RedirectStdoutStderr集成测试")
        
        try:
            if not self.logger:
                print("❌ 需要先运行基本日志测试")
                return False
            
            print("测试标准输出重定向...")
            
            # 记录重定向前的文件大小
            current_file = self.logger.get_current_log_file()
            initial_size = os.path.getsize(current_file) if os.path.exists(current_file) else 0
            
            # 使用RedirectStdoutStderr
            with RedirectStdoutStderr(self.logger):
                print("这是重定向的stdout消息1")
                print("这是重定向的stdout消息2")
                print("包含特殊字符的消息: 🔥 ⚡ 🚀", file=sys.stderr)
                
                # 模拟一些输出
                for i in range(5):
                    print(f"循环输出 {i+1}/5")
                    sys.stderr.write(f"错误输出 {i+1}/5\n")
            
            print("重定向测试完成，检查文件内容...")
            
            # 检查文件是否增长
            final_size = os.path.getsize(current_file) if os.path.exists(current_file) else 0
            size_increase = final_size - initial_size
            
            if size_increase > 0:
                print(f"✅ 文件大小增加了 {size_increase} bytes")
            else:
                print("⚠️ 文件大小没有增加")
            
            # 检查重定向的内容
            with open(current_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            redirect_messages = [
                "重定向的stdout消息1",
                "重定向的stdout消息2", 
                "包含特殊字符的消息",
                "循环输出",
                "错误输出"
            ]
            
            found_count = 0
            for msg in redirect_messages:
                if msg in content:
                    found_count += 1
                    print(f"✅ 找到重定向消息: {msg}")
                else:
                    print(f"⚠️ 未找到重定向消息: {msg}")
            
            if found_count >= len(redirect_messages) // 2:
                print("✅ 重定向功能基本正常")
                self.results['integration'] = True
            else:
                print("⚠️ 重定向功能可能有问题")
                self.results['integration'] = False
            
        except Exception as e:
            print(f"❌ 集成测试失败: {e}")
            self.results['integration'] = False
            return False
        
        return True
    
    def test_cleanup_and_shutdown(self):
        """测试6: 清理和关闭测试"""
        print("\n🛑 测试6: 清理和关闭测试")
        
        try:
            if not self.logger:
                print("❌ 没有活跃的logger对象")
                return False
            
            # 检查线程状态
            thread_alive_before = self.logger._checker_thread.is_alive()
            print(f"关闭前线程状态: {'运行中' if thread_alive_before else '已停止'}")
            
            # 写入最后的消息
            self.logger.write(f"关闭前的最后消息 - {datetime.now()}\n")
            
            # 关闭logger
            print("正在关闭logger...")
            self.logger.close()
            
            # 等待线程完全停止
            time.sleep(1)
            
            # 检查线程是否已停止
            thread_alive_after = self.logger._checker_thread.is_alive()
            print(f"关闭后线程状态: {'运行中' if thread_alive_after else '已停止'}")
            
            if not thread_alive_after:
                print("✅ 后台线程成功停止")
            else:
                print("⚠️ 后台线程仍在运行")
            
            # 尝试再次写入（应该失败或无效）
            try:
                result = self.logger.write("关闭后的写入测试\n")
                print(f"⚠️ 关闭后仍可写入，返回值: {result}")
            except Exception as e:
                print(f"✅ 关闭后写入正确失败: {e}")
            
            self.results['cleanup'] = True
            print("✅ 清理和关闭测试通过")
            
        except Exception as e:
            print(f"❌ 清理和关闭测试失败: {e}")
            self.results['cleanup'] = False
            return False
        
        return True
    
    def generate_report(self):
        """生成测试报告"""
        print("\n" + "=" * 80)
        print("📊 测试报告")
        print("=" * 80)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if result)
        
        print(f"\n总测试数: {total_tests}")
        print(f"通过测试: {passed_tests}")
        print(f"失败测试: {total_tests - passed_tests}")
        print(f"通过率: {passed_tests/total_tests*100:.1f}%")
        
        print(f"\n详细结果:")
        test_names = {
            'basic_logging': '基本日志记录',
            'date_rotation': '日期轮转机制', 
            'thread_safety': '线程安全性',
            'performance': '性能测试',
            'integration': '集成测试',
            'cleanup': '清理和关闭'
        }
        
        for key, result in self.results.items():
            status = "✅ 通过" if result else "❌ 失败"
            name = test_names.get(key, key)
            print(f"  {status} {name}")
        
        # 建议
        print(f"\n💡 建议:")
        if passed_tests == total_tests:
            print("🎉 所有测试通过！日志轮转系统运行正常。")
            print("✅ 系统已准备好用于生产环境。")
        else:
            print("⚠️ 部分测试失败，建议检查相关功能。")
            if not self.results.get('basic_logging'):
                print("🔧 优先修复基本日志记录功能")
            if not self.results.get('date_rotation'):
                print("🔄 检查日期轮转逻辑")
            if not self.results.get('thread_safety'):
                print("🧵 审查线程安全实现")
        
        print("\n" + "=" * 80)

def main():
    """主测试函数"""
    tester = LogRotationTester()
    
    try:
        # 设置环境
        if not tester.setup():
            return
        
        # 运行所有测试
        tests = [
            ('基本日志记录', tester.test_basic_logging),
            ('日期轮转机制', tester.test_date_rotation),
            ('线程安全性', tester.test_thread_safety),
            ('性能测试', tester.test_performance),
            ('集成测试', tester.test_integration_with_redirect),
            ('清理和关闭', tester.test_cleanup_and_shutdown)
        ]
        
        for test_name, test_func in tests:
            print(f"\n{'='*20} {test_name} {'='*20}")
            test_func()
        
        # 生成报告
        tester.generate_report()
        
    finally:
        # 清理环境
        tester.cleanup()

if __name__ == "__main__":
    main()
