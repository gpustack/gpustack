import os
import sys
from datetime import datetime
from threading import Lock, Thread
import time
import logging
from typing import TextIO


class DailyRotatingLogFile:
    """日志文件包装器，支持按日期自动轮转"""
    
    def __init__(self, log_dir: str, encoding: str = "utf-8", check_interval: int = 300):
        """
        初始化日志轮转器
        
        Args:
            log_dir: 日志目录
            encoding: 文件编码
            check_interval: 检查间隔（秒），默认5分钟
        """
        self.log_dir = log_dir
        self.encoding = encoding
        self.check_interval = check_interval
        self._current_file = None
        self._current_date = None
        self._lock = Lock()
        self._stop_thread = False
        self._checker_thread = None
        
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        
        # 立即初始化第一个日志文件
        self._rotate_if_needed()
        
        # 启动后台检查线程
        self._start_checker_thread()
    
    def _start_checker_thread(self):
        """启动后台日期检查线程"""
        self._checker_thread = Thread(target=self._date_checker, daemon=True)
        self._checker_thread.start()
    
    def _date_checker(self):
        """后台线程：每隔指定时间检查日期是否变化"""
        while not self._stop_thread:
            try:
                # 检查是否需要轮转
                self._rotate_if_needed()
                
                # 等待下次检查
                for _ in range(self.check_interval):
                    if self._stop_thread:
                        break
                    time.sleep(1)  # 每秒检查一次停止标志
                    
            except Exception as e:
                # 记录错误但不停止线程
                logging.getLogger("model-manager").error(f"Date checker thread error: {e}")
                time.sleep(60)  # 出错后等待1分钟再继续
    
    def _get_current_date(self):
        return datetime.now().strftime("%Y-%m-%d")
    
    def _get_log_filename(self, date: str):
        return os.path.join(self.log_dir, f"{date}.log")
    
    def _rotate_if_needed(self):
        """如果日期变化，轮转日志文件"""
        current_date = self._get_current_date()
        
        if self._current_date != current_date or self._current_file is None:
            # 记录是否是日期变化（不是首次创建）
            is_date_change = self._current_date is not None and self._current_date != current_date
            
            # 关闭旧文件
            if self._current_file and not self._current_file.closed:
                try:
                    self._current_file.close()
                except Exception:
                    pass  # 忽略关闭文件时的错误
            
            # 打开新的日志文件
            log_filename = self._get_log_filename(current_date)
            try:
                self._current_file = open(log_filename, "a", buffering=1, encoding=self.encoding)
                old_date = self._current_date
                self._current_date = current_date
                
                # 记录日志轮转信息（仅在日期变化时）
                if is_date_change:
                    self._current_file.write(f"\n--- Log rotated from {old_date} to {current_date} ---\n")
                    self._current_file.flush()
            except Exception as e:
                # 如果无法打开新文件，保持旧文件不变
                import logging
                logging.getLogger("model-manager").error(f"Failed to rotate log to {log_filename}: {e}")
                self._current_file = None
    
    def write(self, data: str):
        """写入数据到当前日志文件"""
        with self._lock:
            if self._current_file:
                result = self._current_file.write(data)
                self._current_file.flush()  # 立即刷新，确保实时写入
                return result
            return len(data)  # 返回写入的字符数，模拟标准文件行为
    
    def flush(self):
        with self._lock:
            if self._current_file:
                self._current_file.flush()
    
    def close(self):
        """关闭日志文件和停止后台线程"""
        # 停止后台检查线程
        self._stop_thread = True
        if self._checker_thread and self._checker_thread.is_alive():
            self._checker_thread.join(timeout=2)  # 等待最多2秒
        
        # 关闭文件
        with self._lock:
            if self._current_file and not self._current_file.closed:
                self._current_file.close()
                self._current_file = None
    
    def fileno(self):
        with self._lock:
            self._rotate_if_needed()
            if self._current_file:
                return self._current_file.fileno()
            return None
    
    def readable(self):
        return False
    
    def writable(self):
        return True
    
    def seekable(self):
        return False
    
    def isatty(self):
        return False
    
    def get_current_log_file(self):
        """获取当前日志文件路径，用于调试"""
        with self._lock:
            if self._current_file:
                return self._current_file.name
            # 如果没有当前文件，返回今天应该使用的文件路径
            return self._get_log_filename(self._get_current_date())
