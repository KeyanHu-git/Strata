"""Focused diagnostics tests; no engine, model or device required."""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from serve import console as module
from serve.console import Console


class ConsoleTest(unittest.TestCase):
    def blocked(self, **limits):
        entered, release = threading.Event(), threading.Event()
        output = []

        def write(text):
            entered.set()
            release.wait(5)
            output.append(text)

        console = Console(write, **limits)
        self.addCleanup(lambda: console.close(1))
        self.addCleanup(release.set)
        console.emit('first')
        self.assertTrue(entered.wait(1))
        return console, release, output

    def test_count_and_overload_summary(self):
        console, release, output = self.blocked(max_messages=2)
        self.assertTrue(console.emit('a'))
        self.assertTrue(console.emit('b'))
        for _ in range(100):
            self.assertFalse(console.emit('c'))
        self.assertEqual(console.stats['queued_messages'], 2)
        self.assertEqual(console.stats['dropped'], 100)
        release.set()
        self.assertTrue(console.close(1))
        self.assertEqual(output, ['first', '[console: dropped 100 diagnostics]\n', 'a', 'b'])

    def test_byte_capacity_and_utf8_line_limit(self):
        console, release, output = self.blocked(max_bytes=14, max_line_bytes=8)
        self.assertTrue(console.emit('😀' * 10000))
        self.assertTrue(console.emit('123456'))
        self.assertFalse(console.emit('x'))
        self.assertEqual(console.stats['queued_bytes'], 14)
        release.set()
        console.close(1)
        self.assertIn('😀...\n', output)
        self.assertTrue(all(len(text.encode('utf-8')) <= 8 for text in output))

    def test_progress_slot_identity_and_invalidation(self):
        console, release, output = self.blocked(max_messages=2, max_bytes=20)
        old, new = [], []  # Equal, unhashable keys still have distinct identities.
        console.emit('old', old)
        console.emit('latest', old)
        self.assertEqual(console.stats['queued_messages'], 1)
        self.assertEqual(console.stats['queued_bytes'], 6)
        console.emit('new', new)
        console.discard_progress(old)
        self.assertEqual(console.stats['queued_bytes'], 3)
        console.discard_progress(new)
        self.assertEqual(console.stats['queued_messages'], 0)
        self.assertEqual(console.stats['queued_bytes'], 0)
        console.emit('final', new)
        release.set()
        console.close(1)
        self.assertEqual(output, ['first', 'final'])

    def test_progress_counts_toward_capacity(self):
        console, _, _ = self.blocked(max_messages=1, max_bytes=8)
        key = object()
        self.assertTrue(console.emit('progress', key))
        self.assertFalse(console.emit('normal'))
        self.assertTrue(console.emit('new', key))
        self.assertEqual(console.stats['queued_bytes'], 3)

    def test_blocked_sink_producers_and_close_are_bounded(self):
        console, _, _ = self.blocked()
        worker = console._worker
        done = threading.Event()

        def produce():
            for _ in range(1000):
                console.emit('diagnostic')
            done.set()

        thread = threading.Thread(target=produce, daemon=True)
        thread.start()
        self.assertTrue(done.wait(1))
        start = time.monotonic()
        self.assertFalse(console.close(.02))
        self.assertLess(time.monotonic() - start, .5)
        self.assertFalse(console.emit('after close'))
        self.assertIs(console._worker, worker)
        thread.join(1)

    def test_sink_failure_does_not_kill_writer(self):
        output = []

        def write(text):
            if text == 'bad':
                raise OSError('closed sink')
            output.append(text)

        console = Console(write)
        console.emit('bad')
        console.emit('good')
        self.assertTrue(console.close(1))
        self.assertEqual(output, ['good'])
        self.assertEqual(console.stats['failures'], 1)

    def test_module_injection_and_formatting(self):
        output = []
        console = Console(output.append)
        key = object()
        with mock.patch.object(module, '_console', console):
            module.console_print('a', 2, sep=':', end='!', flush=True)
            module.console_print('b', file=sys.stderr)
            module.progress('p', key)
            module.invalidate_progress(object())
        self.assertTrue(console.close(1))
        self.assertEqual(output, ['a:2!', 'b\n', 'p\n'])

    def test_default_transport_routes_and_never_uses_buffered_methods(self):
        with mock.patch.object(module.os, 'name', 'posix'), \
             mock.patch.object(module.os, 'dup') as dup, \
             mock.patch.object(module.os, 'write', side_effect=lambda fd, b: len(b)) as write, \
             mock.patch.object(module.os, 'close') as close:
            stdout, stderr = mock.Mock(), mock.Mock()
            stdout.fileno.return_value = 1
            stderr.fileno.return_value = 2
            stdout.encoding = stderr.encoding = 'utf-8'
            custom = io.StringIO()
            console = Console(module._standard_write)
            with mock.patch.object(module, '_console', console), \
                 mock.patch.object(sys, 'stdout', stdout), \
                 mock.patch.object(sys, 'stderr', stderr):
                module.console_print('out')
                module.console_print('err', file=sys.stderr)
                module.console_print('custom', file=custom)
                self.assertTrue(console.close(1))
            dup.assert_not_called()
            self.assertEqual([call.args[0] for call in write.call_args_list], [1, 2])
            self.assertEqual([bytes(call.args[1]) for call in write.call_args_list], [b'out\n', b'err\n'])
            close.assert_not_called()
            stdout.write.assert_not_called()
            stderr.write.assert_not_called()
            stdout.flush.assert_not_called()
            stderr.flush.assert_not_called()
            self.assertEqual(custom.getvalue(), 'custom\n')

    def test_publication_captures_redirected_stream(self):
        entered, release = threading.Event(), threading.Event()
        original = module._transport
        captured = io.StringIO()
        console = Console(module._standard_write)
        self.addCleanup(lambda: console.close(1))
        self.addCleanup(release.set)

        def transport(text, stream):
            if text == 'hold':
                entered.set()
                release.wait(2)
                return
            original(text, stream)

        with mock.patch.object(module, '_transport', side_effect=transport), \
             mock.patch.object(module, '_console', console):
            console.emit('hold')
            self.assertTrue(entered.wait(1))
            with redirect_stdout(captured):
                module.console_print('captured')
                module.progress('progress', object())
            release.set()
            self.assertTrue(module.wait_idle(1))
        self.assertEqual(captured.getvalue(), 'captured\nprogress\n')

    def test_wait_idle_timeout_and_later_reuse(self):
        console, release, output = self.blocked()
        start = time.monotonic()
        self.assertFalse(console.wait_idle(.02))
        self.assertLess(time.monotonic() - start, .5)
        release.set()
        self.assertTrue(console.wait_idle(1))
        self.assertTrue(console.emit('after drain'))
        self.assertTrue(console.wait_idle(1))
        self.assertEqual(output, ['first', 'after drain'])

    def test_custom_stream_flush_runs_on_worker(self):
        thread_ids = []
        stream = io.StringIO()
        console = Console(module._standard_write)
        with mock.patch.object(stream, 'flush', side_effect=lambda: thread_ids.append(
                threading.get_ident())), mock.patch.object(module, '_console', console):
            module.console_print('custom', file=stream)
            self.assertTrue(console.wait_idle(1))
        self.assertTrue(console.close(1))
        self.assertEqual(thread_ids, [console._worker.ident])
        self.assertEqual(stream.getvalue(), 'custom\n')

    def test_failed_standard_channel_is_disabled_independently(self):
        console = Console(module._standard_write)
        stdout, stderr = io.StringIO(), io.StringIO()
        attempts = []

        def transport(text, stream):
            attempts.append(text)
            if stream is stdout:
                raise OSError('broken stdout')
            stream.write(text)

        with mock.patch.object(module, '_transport', side_effect=transport), \
             mock.patch.object(module, '_console', console), \
             mock.patch.object(sys, 'stdout', stdout), \
             mock.patch.object(sys, 'stderr', stderr):
            module.console_print('fail')
            self.assertTrue(console.wait_idle(1))
            for _ in range(20):
                module.console_print('suppressed')
            module.console_print('healthy', file=stderr)
            self.assertTrue(console.close(1))
        self.assertEqual(attempts, ['fail\n', 'healthy\n'])
        self.assertEqual(console.stats['failures'], 1)
        self.assertEqual(console.stats['dropped'], 20)
        self.assertEqual(stderr.getvalue(), 'healthy\n')

    def test_truncation_marker_newline_and_no_double_progress_newline(self):
        output = []
        console = Console(output.append, max_line_bytes=12)
        with mock.patch.object(module, '_console', console):
            module.console_print('12345678901234567890')
            module.console_print('next')
            module.progress('progress\n', object())
        self.assertTrue(console.close(1))
        self.assertEqual(output, ['12345678...\n', 'next\n', 'progress\n'])
        for limit in range(1, 5):
            small = Console(lambda text: None, max_line_bytes=limit)
            self.assertEqual(small._bounded('x' * 10), '.' * (limit - 1) + '\n')
            self.assertTrue(small.close(1))

    def test_transport_borrows_fd_without_closing_or_duplicating(self):
        with tempfile.TemporaryFile(mode='w+b') as stream:
            with mock.patch.object(module.os, 'dup') as dup:
                module._transport('borrowed', stream)
            dup.assert_not_called()
            os.fstat(stream.fileno())
            stream.seek(0)
            self.assertEqual(stream.read(), b'borrowed')

    def test_pipe_newlines_match_default_text_wrapper(self):
        outputs = []
        for raw in (False, True):
            reader_fd, writer_fd = os.pipe()
            with os.fdopen(reader_fd, 'rb') as reader:
                with os.fdopen(writer_fd, 'w', encoding='utf-8') as writer:
                    if raw:
                        module._transport('alpha\nbeta\n', writer)
                    else:
                        writer.write('alpha\nbeta\n')
                outputs.append(reader.read())
        newline = b'\r\n' if os.name == 'nt' else b'\n'
        self.assertEqual(outputs[0], b'alpha' + newline + b'beta' + newline)
        self.assertEqual(outputs[1], outputs[0])

    def test_transport_handles_partial_writes(self):
        stream = mock.Mock()
        stream.fileno.return_value = 7
        stream.encoding = 'ascii'
        chunks = []

        def write(fd, data):
            self.assertEqual(fd, 7)
            chunks.append(bytes(data[:1]))
            return 1

        with mock.patch.object(module.os, 'name', 'posix'), \
             mock.patch.object(module.os, 'write', side_effect=write), \
             mock.patch.object(module.os, 'close') as close:
            module._transport('aé!', stream)
        self.assertEqual(b''.join(chunks), b'a?!')
        close.assert_not_called()
        stream.write.assert_not_called()

    def test_windows_console_transport_uses_utf8_and_handles_partial_writes(self):
        stream = mock.Mock()
        stream.fileno.return_value = 7
        stream.encoding = 'ascii'  # Console's raw API requires UTF-8 regardless.
        raw = mock.Mock()
        chunks = []

        def write(data):
            chunks.append(bytes(data[:1]))
            return 1

        raw.write.side_effect = write
        with mock.patch.object(module.os, 'name', 'nt'), \
             mock.patch.object(module, 'open', return_value=raw, create=True) as opened, \
             mock.patch.object(module.os, 'write') as fd_write:
            module._transport('aé中!', stream)
        opened.assert_called_once_with(7, 'wb', buffering=0, closefd=False)
        self.assertEqual(b''.join(chunks), 'aé中!'.encode('utf-8'))
        raw.close.assert_called_once_with()
        fd_write.assert_not_called()
        stream.write.assert_not_called()
        stream.flush.assert_not_called()

    def test_windows_redirected_file_preserves_stream_encoding_and_fd(self):
        with tempfile.TemporaryFile(mode='w+b') as stream:
            destination = mock.Mock()
            destination.fileno.return_value = stream.fileno()
            destination.encoding = 'utf-16-le'
            with mock.patch.object(module.os, 'name', 'nt'):
                module._transport('é中\n', destination)
            os.fstat(stream.fileno())
            stream.seek(0)
            self.assertEqual(stream.read(), 'é中\r\n'.encode('utf-16-le'))

    def test_transport_releases_wrapper_on_failure_without_closing_fd(self):
        for result in (0, None, OSError('broken sink')):
            with self.subTest(result=result):
                raw = mock.Mock()
                if isinstance(result, Exception):
                    raw.write.side_effect = result
                else:
                    raw.write.return_value = result
                stream = mock.Mock(encoding='utf-8')
                with mock.patch.object(module.os, 'name', 'nt'), \
                     mock.patch.object(module, 'open', return_value=raw, create=True), \
                     mock.patch.object(module.os, 'close') as close:
                    with self.assertRaises(OSError):
                        module._transport('text', stream)
                raw.close.assert_called_once_with()
                close.assert_not_called()

    def test_failed_summary_is_reported_after_recovery(self):
        entered, release = threading.Event(), threading.Event()
        output = []
        failed = False

        def write(text):
            nonlocal failed
            if text == 'first':
                entered.set()
                release.wait(2)
            if text.startswith('[console:') and not failed:
                failed = True
                raise OSError('temporary failure')
            output.append(text)

        console = Console(write, max_messages=1)
        self.addCleanup(lambda: console.close(1))
        self.addCleanup(release.set)
        console.emit('first')
        self.assertTrue(entered.wait(1))
        console.emit('next')
        self.assertFalse(console.emit('dropped'))
        release.set()
        self.assertTrue(console.close(1))
        self.assertEqual(output, ['first', 'next', '[console: dropped 1 diagnostics]\n'])
        self.assertEqual(console.stats['failures'], 1)

    def test_invalid_limits(self):
        for limits in ({'max_messages': 0}, {'max_bytes': -1},
                       {'max_line_bytes': 1.5}):
            with self.assertRaises(ValueError):
                Console(lambda text: None, **limits)

    def test_unavailable_standard_stream_only_fails_delivery(self):
        console = Console(module._standard_write)
        with mock.patch.object(sys, 'stdout', None):
            console.emit('lost')
            self.assertTrue(console.close(1))
        self.assertEqual(console.stats['failures'], 1)

    def test_lazy_singleton_and_shutdown_no_replacement(self):
        with mock.patch.object(module, '_console', None), \
             mock.patch.object(module, '_shutdown', False):
            consoles = []
            threads = [threading.Thread(target=lambda: consoles.append(module._get_console()))
                       for _ in range(16)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(1)
            self.assertEqual(len(consoles), 16)
            self.assertTrue(all(console is consoles[0] for console in consoles))
            self.assertTrue(module.shutdown(1))
            self.assertIs(module._get_console(), consoles[0])
            self.assertFalse(consoles[0].emit('closed'))

    @unittest.skipUnless(os.name == 'nt', 'Windows console API')
    def test_real_windows_console_unicode_readback(self):
        # A fresh, hidden console isolates code pages and screen contents from
        # the user's terminal. Pipes only carry the child's assertion result.
        script = r'''
import ctypes
from ctypes import wintypes
import io
import msvcrt
import os
from serve import console

class COORD(ctypes.Structure):
    _fields_ = [('X', wintypes.SHORT), ('Y', wintypes.SHORT)]

class SCREEN_INFO(ctypes.Structure):
    _fields_ = [('size', COORD), ('position', COORD), ('attributes', wintypes.WORD),
                ('window', wintypes.SMALL_RECT), ('maximum', COORD)]

kernel = ctypes.WinDLL('kernel32', use_last_error=True)
get_info = kernel.GetConsoleScreenBufferInfo
get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(SCREEN_INFO)]
get_info.restype = wintypes.BOOL
create = kernel.CreateConsoleScreenBuffer
create.argtypes = [wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                   wintypes.DWORD, ctypes.c_void_p]
create.restype = wintypes.HANDLE
read = kernel.ReadConsoleOutputCharacterW
read.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD,
                 COORD, ctypes.POINTER(wintypes.DWORD)]
read.restype = wintypes.BOOL
cursor = kernel.SetConsoleCursorPosition
cursor.argtypes = [wintypes.HANDLE, COORD]
cursor.restype = wintypes.BOOL
set_cp = kernel.SetConsoleOutputCP
set_cp.argtypes = [wintypes.UINT]
set_cp.restype = wintypes.BOOL
close = kernel.CloseHandle
close.argtypes = [wintypes.HANDLE]
close.restype = wintypes.BOOL

def checked(ok):
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())

handle = create(0xC0000000, 3, None, 1, None)
if handle == ctypes.c_void_p(-1).value:
    raise ctypes.WinError(ctypes.get_last_error())
try:
    fd = msvcrt.open_osfhandle(handle, os.O_BINARY | os.O_RDWR)
except BaseException:
    close(handle)
    raise
# fd now owns handle; both the stream and borrowed transports use this handle.
with os.fdopen(fd, 'w', encoding='utf-8') as stream:
    with open(fd, 'wb', buffering=0, closefd=False) as raw:
        assert not isinstance(raw, io.FileIO), type(raw)
    expected = 'caf\u00e9 \u4e2d\u6587 \u03a9 \u0416'
    for codepage in (437, 936):
        checked(set_cp(codepage))
        checked(cursor(handle, COORD(0, 0)))
        console.console_print(expected, end='', file=stream)
        assert console.wait_idle(2), 'console did not drain'
        assert console._console.stats['failures'] == 0
        # CJK glyphs occupy two screen cells; read beyond the text and strip
        # only the untouched trailing spaces in this fresh screen buffer.
        cells = len(expected) * 2
        actual = ctypes.create_unicode_buffer(cells + 1)
        count = wintypes.DWORD()
        checked(read(handle, actual, cells, COORD(0, 0), ctypes.byref(count)))
        assert count.value > 0, count.value
        assert actual.value.rstrip(' ') == expected, (codepage, ascii(actual.value), ascii(expected))
        layouts = []
        for start_row, use_transport in ((3, False), (7, True)):
            checked(cursor(handle, COORD(0, start_row)))
            if use_transport:
                console.console_print('alpha\nbeta', end='', file=stream)
                assert console.wait_idle(2), 'multiline output did not drain'
                assert console._console.stats['failures'] == 0
            else:
                stream.write('alpha\nbeta')
                stream.flush()
            info = SCREEN_INFO()
            checked(get_info(handle, ctypes.byref(info)))
            lines = []
            for row in (start_row, start_row + 1):
                line = ctypes.create_unicode_buffer(17)
                checked(read(handle, line, 16, COORD(0, row), ctypes.byref(count)))
                lines.append(line.value.rstrip(' '))
            layouts.append((lines, info.position.X, info.position.Y - start_row))
        expected_layout = (['alpha', 'beta'], 4, 1)
        assert layouts == [expected_layout, expected_layout], (codepage, layouts)
    assert console.shutdown(1)
print('console Unicode readback passed: CP437 and CP936; multiline matches TextIOWrapper, cursor=(4,1)')
'''
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = subprocess.SW_HIDE
        env = os.environ.copy()
        env.pop('PYTHONLEGACYWINDOWSSTDIO', None)
        result = subprocess.run(
            [sys.executable, '-B', '-c', script],
            cwd=Path(__file__).resolve().parents[1], capture_output=True,
            timeout=15, creationflags=subprocess.CREATE_NEW_CONSOLE,
            startupinfo=startup, env=env)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
        self.assertIn(b'console Unicode readback passed: CP437 and CP936', result.stdout)

    @unittest.skipUnless(hasattr(os, 'fork'), 'fork unavailable')
    def test_fork_child_has_fresh_singleton(self):
        script = r'''
import os
from serve import console
parent = console._get_console()
pid = os.fork()
if pid == 0:
    child = console._get_console()
    valid = child is not parent and child._worker.is_alive()
    valid = console.shutdown(1) and valid
    os._exit(0 if valid else 1)
_, status = os.waitpid(pid, 0)
assert status == 0
assert console._get_console() is parent
assert console.shutdown(1)
'''
        result = subprocess.run([sys.executable, '-c', script],
                                cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr.decode())

    @unittest.skipUnless(os.name == 'posix', 'POSIX pipe saturation')
    def test_real_full_pipe_subprocess_exits(self):
        script = r'''
import os, threading
from serve import console
r, w = os.pipe()
os.set_blocking(w, False)
try:
    while True:
        os.write(w, b'x' * 4096)
except BlockingIOError:
    pass
os.set_blocking(w, True)
os.dup2(w, 1)
os.close(w)
entered = threading.Event()
original = os.write
def write(fd, data):
    entered.set()
    return original(fd, data)
os.write = write
console.console_print('blocked diagnostic')
assert entered.wait(1)
assert not console.shutdown(.02)
# Keep the pipe read end open and unread through interpreter finalization.
'''
        result = subprocess.run([sys.executable, '-c', script],
                                cwd=Path(__file__).resolve().parents[1],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr.decode())


if __name__ == '__main__':
    unittest.main()
