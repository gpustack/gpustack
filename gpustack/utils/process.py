import psutil


def terminate_process_tree(pid: int):
    process = psutil.Process(pid)
    children = process.children(recursive=True)
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(children, timeout=3)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        process.terminate()
        process.wait(timeout=3)
    except psutil.TimeoutExpired:
        process.kill()
