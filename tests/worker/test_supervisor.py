import errno
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import textwrap
import time

import psutil
import pytest

SUPERVISOR = (
    Path(__file__).resolve().parents[2] / "gpustack/worker/backends/supervisor.py"
)
pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="Requires POSIX process groups"
)


def _wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate(), "Timed out waiting for the child process"


def _stopped(pid):
    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def _task(tmp_path, name, code, env=None):
    prelude = (
        "import os\nfrom pathlib import Path\n"
        f"Path({str(tmp_path / (name + '.pid'))!r}).write_text(str(os.getpid()))\n"
    )
    return {
        "name": name,
        "argv": [sys.executable, "-u", "-c", prelude + textwrap.dedent(code)],
        "env": env or {},
    }


@pytest.fixture
def start_supervisor(tmp_path):
    processes = []

    def start(tasks, shutdown_timeout=1, subreaper=False):
        if not hasattr(os, 'waitid'):
            pytest.skip('Requires non-reaping waitid')
        config = tmp_path / "tasks.json"
        config.write_text(
            json.dumps(
                {
                    "tasks": tasks,
                    "shutdown_timeout": shutdown_timeout,
                    "cleanup_timeout": 1,
                }
            )
        )
        command = [sys.executable, "-I", "-S", str(SUPERVISOR), str(config)]
        if subreaper:
            command = [
                sys.executable,
                "-I",
                "-S",
                "-c",
                "import ctypes,runpy,sys;"
                "PR_SET_CHILD_SUBREAPER=36;"
                "assert ctypes.CDLL(None).prctl(PR_SET_CHILD_SUBREAPER,1,0,0,0)==0;"
                f"sys.argv=[{str(SUPERVISOR)!r},{str(config)!r}];"
                f"runpy.run_path({str(SUPERVISOR)!r},run_name='__main__')",
            ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes.append(process)
        return process

    yield start

    for process in processes:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        process.stdout.close()
    for path in tmp_path.glob("*.pid"):
        if path.read_text():
            try:
                os.killpg(int(path.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_all_tasks_start_with_their_own_device_environment(
    start_supervisor, tmp_path, monkeypatch
):
    monkeypatch.setenv("SUPERVISOR_INHERITED", "present")
    tasks = []
    for rank in range(2):
        tasks.append(
            _task(
                tmp_path,
                f"dp{rank}-node0",
                f"""
                import json, time
                Path({str(tmp_path / f'ready-{rank}')!r}).touch()
                while not Path({str(tmp_path / f'ready-{1 - rank}')!r}).exists():
                    time.sleep(0.01)
                print(json.dumps([os.environ['CUDA_VISIBLE_DEVICES'],
                                  os.environ['ASCEND_RT_VISIBLE_DEVICES'],
                                  os.environ['SUPERVISOR_INHERITED']]), flush=True)
                Path({str(tmp_path / f'initialized-{rank}')!r}).touch()
                time.sleep(60)
                """,
                env={
                    "CUDA_VISIBLE_DEVICES": str(rank),
                    "ASCEND_RT_VISIBLE_DEVICES": str(rank + 2),
                },
            )
        )
    process = start_supervisor(tasks)
    _wait_for(
        lambda: all((tmp_path / f"initialized-{rank}").exists() for rank in range(2))
    )
    process.terminate()
    output = process.communicate(timeout=5)[0].decode()
    assert process.returncode == 128 + signal.SIGTERM, output
    assert "[gpustack-supervisor] started dp0-node0" in output
    for rank in range(2):
        assert f'[dp{rank}-node0] ["{rank}", "{rank + 2}", "present"]' in output
        pid = int((tmp_path / f"dp{rank}-node0.pid").read_text())
        assert not psutil.pid_exists(pid)


@pytest.mark.parametrize("exit_code", [0, 7])
def test_any_exit_stops_all_peers_with_one_shutdown_deadline(
    start_supervisor, tmp_path, exit_code
):
    tasks = []
    for rank in range(4):
        tasks.append(
            _task(
                tmp_path,
                f"dp{rank}-node0",
                f"""
                import signal, time
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                Path({str(tmp_path / f'ready-{rank}')!r}).touch()
                time.sleep(60)
                """,
            )
        )
    tasks.append(
        _task(
            tmp_path,
            "dp4-node0",
            f"""
            import sys, time
            while not all(Path({str(tmp_path)!r}, f'ready-{{rank}}').exists() for rank in range(4)):
                time.sleep(0.01)
            sys.exit({exit_code})
            """,
        )
    )
    started = time.monotonic()
    process = start_supervisor(tasks, shutdown_timeout=0.8)
    output = process.communicate(timeout=5)[0].decode()
    assert process.returncode == 1, output
    assert time.monotonic() - started < 2.5
    assert "dp4-node0" in output and f"exited with code {exit_code}" in output
    for path in tmp_path.glob("*.pid"):
        assert not psutil.pid_exists(int(path.read_text()))


def test_partial_spawn_failure_cleans_up_started_processes(start_supervisor, tmp_path):
    tasks = [
        _task(tmp_path, "dp0-node0", "import time; time.sleep(60)"),
        {"name": "dp1-node0", "argv": [str(tmp_path / "missing-command")]},
    ]
    process = start_supervisor(tasks)
    output = process.communicate(timeout=5)[0].decode()
    assert process.returncode == 1
    assert "dp1-node0" in output
    pids = re.findall(r"started dp0-node0 pid=(\d+)", output)
    assert len(pids) == 1, output
    assert not psutil.pid_exists(int(pids[0]))


@pytest.mark.parametrize("stop_signal", [signal.SIGTERM, signal.SIGINT])
def test_stop_signal_reaches_the_whole_task_process_group(
    start_supervisor, tmp_path, stop_signal
):
    child_code = f"""
import signal, time
from pathlib import Path
def stop(signum, frame):
    Path({str(tmp_path / 'child-signal')!r}).write_text(str(signum))
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
Path({str(tmp_path / 'child-ready')!r}).touch()
time.sleep(60)
"""
    task = _task(
        tmp_path,
        "dp0-node1",
        f"""
        import signal, subprocess, sys, time
        def stop(signum, frame):
            Path({str(tmp_path / 'parent-signal')!r}).write_text(str(signum))
            child.wait(timeout=2)
            raise SystemExit(0)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        child = subprocess.Popen([sys.executable, '-c', {child_code!r}])
        Path({str(tmp_path / 'child.pid')!r}).write_text(str(child.pid))
        time.sleep(60)
        """,
    )
    process = start_supervisor([task])
    _wait_for(lambda: (tmp_path / "child-ready").exists())
    process.send_signal(stop_signal)
    output = process.communicate(timeout=5)[0].decode()
    assert process.returncode == 128 + stop_signal, output
    assert (tmp_path / "parent-signal").read_text() == str(stop_signal)
    assert (tmp_path / "child-signal").read_text() == str(stop_signal)
    for path in tmp_path.glob("*.pid"):
        assert not psutil.pid_exists(int(path.read_text()))


@pytest.mark.parametrize(
    "subreaper",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                sys.platform != "linux", reason="Requires Linux child-subreaper support"
            ),
        ),
    ],
)
def test_cleanup_kills_descendants_after_the_group_leader_exits(
    start_supervisor, tmp_path, subreaper
):
    child_code = f"""
import signal, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path({str(tmp_path / 'child-ready')!r}).touch()
time.sleep(60)
"""
    task = _task(
        tmp_path,
        "dp0-node0",
        f"""
        import subprocess, sys, time
        child = subprocess.Popen([sys.executable, '-c', {child_code!r}])
        Path({str(tmp_path / 'child.pid')!r}).write_text(str(child.pid))
        while not Path({str(tmp_path / 'child-ready')!r}).exists():
            time.sleep(0.01)
        """,
    )
    process = start_supervisor([task], shutdown_timeout=0.5, subreaper=subreaper)
    output = process.communicate(timeout=5)[0].decode()
    assert process.returncode == 1, output
    child_pid = int((tmp_path / "child.pid").read_text())
    _wait_for(lambda: _stopped(child_pid))
    if subreaper:
        assert not psutil.pid_exists(child_pid)
    assert not psutil.pid_exists(int((tmp_path / "dp0-node0.pid").read_text()))


def test_long_output_is_tagged_and_split_into_bounded_fragments(
    start_supervisor, tmp_path
):
    task = _task(
        tmp_path,
        "dp0-node0",
        "os.write(1, ('汉' * 20000).encode() + b'\\n'); os.write(2, b'stderr-line\\n')",
    )
    process = start_supervisor([task])
    output = process.communicate(timeout=5)[0].decode()
    task_lines = [
        line for line in output.splitlines() if line.startswith("[dp0-node0]")
    ]
    assert len(task_lines) > 1
    assert all(len(line) <= 4200 for line in task_lines)
    assert "�" not in output
    assert "[dp0-node0] stderr-line" in output


def test_final_log_fragment_is_emitted_without_a_newline(start_supervisor, tmp_path):
    process = start_supervisor(
        [_task(tmp_path, "dp0-node0", "os.write(1, b'last-fragment')")]
    )
    output = process.communicate(timeout=5)[0].decode()
    assert process.returncode == 1, output
    assert "[dp0-node0] last-fragment\n" in output


def test_blocked_stdout_does_not_delay_fault_detection_or_cleanup(
    start_supervisor, tmp_path
):
    flood = _task(
        tmp_path,
        "dp0-node0",
        f"""
        Path({str(tmp_path / 'flood-ready')!r}).touch()
        while True:
            os.write(1, b'x' * 65536)
        """,
    )
    failure = _task(
        tmp_path,
        "dp1-node0",
        f"""
        import sys, time
        while not Path({str(tmp_path / 'flood-ready')!r}).exists():
            time.sleep(0.01)
        time.sleep(0.2)
        sys.exit(9)
        """,
    )
    process = start_supervisor([flood, failure], shutdown_timeout=0.5)
    assert process.wait(timeout=4) == 1
    process.communicate(timeout=2)
    for path in tmp_path.glob("*.pid"):
        assert not psutil.pid_exists(int(path.read_text()))


@pytest.mark.parametrize(
    "tasks",
    [
        [],
        [{"name": "dp0-node0", "argv": "echo unsafe"}],
        [{"name": "dp0-node0", "argv": ["echo"], "env": {"CUDA_VISIBLE_DEVICES": 0}}],
    ],
)
def test_invalid_configuration_starts_no_tasks(start_supervisor, tmp_path, tasks):
    process = start_supervisor(tasks)
    output = process.communicate(timeout=5)[0].decode()
    assert process.returncode != 0, output
    assert "Invalid supervisor configuration" in output
    assert not list(tmp_path.glob("*.pid"))


def test_all_configuration_is_validated_before_the_first_spawn(
    start_supervisor, tmp_path
):
    process = start_supervisor(
        [
            _task(tmp_path, "dp0-node0", "import time; time.sleep(60)"),
            {"name": "dp1-node0", "argv": []},
        ]
    )
    output = process.communicate(timeout=5)[0].decode()
    assert "Invalid supervisor configuration" in output
    assert process.returncode != 0
    assert not list(tmp_path.glob("*.pid"))


def test_emit_drains_available_output_before_dropping_a_new_fragment(monkeypatch):
    from gpustack.worker.backends import supervisor

    runner = supervisor.GPUStackSupervisor([], 1)
    try:
        runner.output.extend(b'x' * supervisor._LOG_BUFFER_SIZE)
        written = []

        def write(fd, block):
            written.append(bytes(block))
            return len(block)

        monkeypatch.setattr(supervisor.os, 'write', write)
        runner._emit('rank', 'new')
        assert b''.join(written) == b'x' * supervisor._LOG_BUFFER_SIZE + b'[rank] new\n'
        assert runner.dropped_logs == 0
        assert not runner.output
    finally:
        runner.selector.close()


def test_missing_waitid_is_rejected_before_loading_or_starting_tasks(
    monkeypatch, capsys
):
    from unittest.mock import Mock
    from gpustack.worker.backends import supervisor

    monkeypatch.delattr(supervisor.os, 'waitid', raising=False)
    monkeypatch.setattr(sys, 'argv', ['gpustack-supervisor', 'unused.json'])
    load = Mock()
    start = Mock()
    monkeypatch.setattr(supervisor, '_load_config', load)
    monkeypatch.setattr(supervisor, 'GPUStackSupervisor', start)

    with pytest.raises(SystemExit) as error:
        supervisor.main()

    assert error.value.code == 2
    assert 'Process supervision requires os.waitid' in capsys.readouterr().err
    load.assert_not_called()
    start.assert_not_called()


@pytest.mark.parametrize('error_number', [errno.EIO, errno.EPIPE])
def test_log_write_error_retains_bounded_output_for_retry(monkeypatch, error_number):
    from unittest.mock import Mock
    from gpustack.worker.backends import supervisor

    runner = supervisor.GPUStackSupervisor([], 1)
    try:
        runner.output.extend(b'[rank] important\n')
        write = Mock(side_effect=OSError(error_number, 'write failed'))
        monkeypatch.setattr(supervisor.os, 'write', write)
        runner._flush_logs()
        assert runner.output == b'[rank] important\n'
        write.side_effect = None
        write.return_value = len(runner.output)
        runner._flush_logs()
        assert not runner.output
        assert runner.dropped_logs == 0
    finally:
        runner.selector.close()


@pytest.mark.skipif(not hasattr(os, 'waitid'), reason='Requires non-reaping waitid')
def test_fault_monitor_reserves_leader_pid_until_all_group_signals_finish(
    tmp_path,
    monkeypatch,
):
    from gpustack.worker.backends import supervisor

    task = supervisor._parse_task(_task(tmp_path, 'rank', 'raise SystemExit(9)'))
    runner = supervisor.GPUStackSupervisor([task], 0.1)
    runner._emit = lambda *args: None
    real_signal = runner._signal_groups
    sent = []

    def send(signum):
        assert task.process.returncode is None
        status = os.waitid(os.P_PID, task.process.pid, os.WEXITED | os.WNOWAIT)
        assert status.si_status == 9
        sent.append(signum)
        real_signal(signum)

    try:
        assert runner._start_tasks()
        assert runner._monitor() == 1
        monkeypatch.setattr(runner, '_signal_groups', send)
        assert runner._cleanup()
        assert sent == [signal.SIGTERM, signal.SIGKILL]
        assert task.process.returncode == 9
    finally:
        if task.process is not None:
            task.process.wait(timeout=2)
            if task.process.stdout is not None:
                task.process.stdout.close()
        runner.selector.close()


def test_confirmed_dead_group_is_never_probed_or_signalled_again(monkeypatch):
    from unittest.mock import Mock
    from gpustack.worker.backends import supervisor

    task = supervisor._parse_task({'name': 'rank', 'argv': ['unused']})
    task.process = Mock(pid=12345, returncode=0)
    runner = supervisor.GPUStackSupervisor([task], 1)
    kill = Mock(side_effect=ProcessLookupError())
    monkeypatch.setattr(supervisor.os, 'killpg', kill)
    monkeypatch.setattr(supervisor.os, 'waitpid', Mock(side_effect=ChildProcessError()))
    try:
        assert not runner._groups_alive()
        kill.reset_mock(side_effect=True)
        assert not runner._groups_alive()
        runner._signal_groups(signal.SIGKILL)
        kill.assert_not_called()
    finally:
        runner.selector.close()


def test_shutdown_uses_full_grace_and_one_shared_cleanup_deadline(monkeypatch):
    from unittest.mock import Mock
    from gpustack.worker.backends import supervisor

    tasks = [
        supervisor._parse_task({'name': f'rank{i}', 'argv': ['unused']})
        for i in range(2)
    ]
    runner = supervisor.GPUStackSupervisor(tasks, 60, cleanup_timeout=5)
    now, signals, waits = [100.0], [], []

    def wait(timeout):
        waits.append(timeout)
        now[0] += timeout / 2

    for task in tasks:
        task.process = Mock(returncode=None)
        task.process.wait.side_effect = wait
    monkeypatch.setattr(supervisor.time, 'monotonic', lambda: now[0])
    runner._signal_groups = lambda sig: signals.append((sig, now[0]))
    runner._groups_alive = lambda: len(signals) < 2
    runner._read_logs = lambda timeout: now.__setitem__(0, now[0] + timeout)
    runner._emit = Mock()
    try:
        assert runner._cleanup()
        assert signals == [(signal.SIGTERM, 100.0), (signal.SIGKILL, 160.0)]
        assert waits == [5, 2.5]
        assert now[0] < 165
    finally:
        runner.selector.close()


def test_cooperative_shutdown_finishes_within_grace(start_supervisor, tmp_path):
    done = tmp_path / 'drained'
    task = _task(
        tmp_path,
        'cooperative',
        f'''
        import signal, time
        def stop(_signal, _frame):
            time.sleep(0.25)
            Path({str(done)!r}).touch()
            raise SystemExit(0)
        signal.signal(signal.SIGTERM, stop)
        Path({str(tmp_path / 'ready')!r}).touch()
        while True:
            time.sleep(0.01)
        ''',
    )
    process = start_supervisor([task], shutdown_timeout=0.3)
    _wait_for(lambda: (tmp_path / 'ready').exists())
    process.terminate()
    output = process.communicate(timeout=5)[0].decode()
    assert process.returncode == 128 + signal.SIGTERM, output
    assert done.exists()
    assert not psutil.pid_exists(int((tmp_path / 'cooperative.pid').read_text()))
