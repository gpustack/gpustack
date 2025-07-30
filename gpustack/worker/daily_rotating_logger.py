import os
import sys
from datetime import datetime
from threading import Lock
from typing import TextIO


class DailyRotatingLogFile:
    """日志文件包装器，支持按日期自动轮转"""
    
    def __init__(self, log_dir: str, encoding: str = "utf-8"):
        self.log_dir = log_dir
        self.encoding = encoding
        self._current_file = None
        self._current_date = None
        self._lock = Lock()
        
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        
        # 立即初始化第一个日志文件
        self._rotate_if_needed()
    
    def _get_current_date(self):
        return datetime.now().strftime("%Y-%m-%d")
    
    def _get_log_filename(self, date: str):
        return os.path.join(self.log_dir, f"{date}.log")
    
    def _rotate_if_needed(self):
        """如果日期变化，轮转日志文件"""
        current_date = self._get_current_date()
        
        if self._current_date != current_date:
            if self._current_file and not self._current_file.closed:
                try:
                    self._current_file.close()
                except Exception:
                    pass  # 忽略关闭文件时的错误
            
            log_filename = self._get_log_filename(current_date)
            try:
                self._current_file = open(log_filename, "a", buffering=1, encoding=self.encoding)
                self._current_date = current_date
            except Exception as e:
                # 如果无法打开新文件，保持旧文件不变
                pass
    
    def write(self, data: str):
        with self._lock:
            self._rotate_if_needed()
            if self._current_file:
                self._current_file.write(data)
                self._current_file.flush()  # 立即刷新，确保实时写入
            return len(data)  # 返回写入的字符数，模拟标准文件行为
    
    def flush(self):
        with self._lock:
            if self._current_file:
                self._current_file.flush()
    
    def close(self):
        with self._lock:
            if self._current_file and not self._current_file.closed:
                self._current_file.close()
    
    def fileno(self):
        with self._lock:
            self._rotate_if_needed()
            if self._current_file:
                return self._current_file.fileno()
            return None
