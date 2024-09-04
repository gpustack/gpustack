import os

import psutil


def signal_handler(signum, frame):
    pid = os.getpid()
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for process in children:
        process.send_signal(signum)
    os._exit(0)
