import threading
import unittest

from gpustack.utils.task import run_periodically_in_thread


class ObservedEvent(threading.Event):
    def __init__(self):
        super().__init__()
        self.waiting = threading.Event()

    def wait(self, timeout=None):
        self.waiting.set()
        return super().wait(timeout)


class TestPeriodicStop(unittest.TestCase):
    def test_running_callback_finishes_before_thread_exits(self):
        stop = threading.Event()
        entered = threading.Event()
        release = threading.Event()

        def check():
            entered.set()
            release.wait()

        thread = run_periodically_in_thread(check, 0.5, 0, stop)
        try:
            self.assertTrue(entered.wait(1))
            stop.set()
            thread.join(0.05)
            self.assertTrue(thread.is_alive())
            release.set()
            thread.join(0.15)
            self.assertFalse(thread.is_alive())
        finally:
            stop.set()
            release.set()
            thread.join(1)
            self.assertFalse(thread.is_alive())

    def test_stop_during_initial_delay(self):
        stop = ObservedEvent()
        called = threading.Event()
        thread = run_periodically_in_thread(called.set, 0.5, 0.5, stop)
        try:
            # Waiting is observed when supported; the bounded wait also lets
            # the unmodified implementation reach its initial sleep.
            stop.waiting.wait(0.05)
            stop.set()
            thread.join(0.15)
            self.assertFalse(thread.is_alive())
            self.assertFalse(called.is_set())
        finally:
            stop.set()
            thread.join(1)
            self.assertFalse(thread.is_alive())

    def test_stop_between_calls(self):
        stop = ObservedEvent()
        called = threading.Event()
        thread = run_periodically_in_thread(called.set, 0.5, 0, stop)
        try:
            self.assertTrue(called.wait(1))
            stop.waiting.wait(0.05)
            stop.set()
            thread.join(0.15)
            self.assertFalse(thread.is_alive())
        finally:
            stop.set()
            thread.join(1)
            self.assertFalse(thread.is_alive())

    def test_pre_stopped_with_initial_delay(self):
        stop = threading.Event()
        stop.set()
        called = threading.Event()
        thread = run_periodically_in_thread(called.set, 0.5, 0.5, stop)
        try:
            thread.join(0.15)
            self.assertFalse(thread.is_alive())
            self.assertFalse(called.is_set())
        finally:
            thread.join(1)
            self.assertFalse(thread.is_alive())

    def test_arguments_and_retry_after_exception(self):
        stop = threading.Event()
        calls = []

        def check(value, *, marker):
            calls.append((value, marker))
            if len(calls) == 1:
                raise RuntimeError('retry')
            stop.set()

        thread = run_periodically_in_thread(check, 0.01, 0, stop, 7, marker='ok')
        try:
            thread.join(1)
            self.assertFalse(thread.is_alive())
            self.assertEqual(calls, [(7, 'ok'), (7, 'ok')])
        finally:
            stop.set()
            thread.join(1)
