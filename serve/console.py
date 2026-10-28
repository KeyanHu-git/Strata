"""Best-effort diagnostics: bounded pending storage and one daemon writer.

Capacity includes the single coalesced progress slot, but excludes the one
in-flight record. Keys are compared by identity, never mutated or hashed.
Invalidation cannot retract a record already handed to the sink. close() stops
admission and drains pending records, waiting at most timeout seconds.
"""
import atexit
from collections import deque
import io
import os
import sys
import threading


def _transport(text, destination):
    stream = destination
    if stream is None:
        raise OSError("diagnostic stream unavailable")
    try:
        fd = stream.fileno()
    except (AttributeError, OSError, ValueError):
        stream.write(text)
        stream.flush()
        return
    # Borrow the caller's FD through the drain. Windows open selects its
    # Unicode console transport (UTF-8 -> WriteConsoleW) or ordinary FileIO.
    raw = open(fd, "wb", buffering=0, closefd=False) if os.name == "nt" else None
    try:
        if os.name == "nt":
            text = text.replace("\n", "\r\n")
        encoding = getattr(stream, "encoding", None) or "utf-8"
        if raw is not None and not isinstance(raw, io.FileIO):
            encoding = "utf-8"
        try:
            data = text.encode(encoding, errors="replace")
        except LookupError:
            data = text.encode("utf-8", errors="replace")
        view = memoryview(data)
        while view:
            written = raw.write(view) if raw is not None else os.write(fd, view)
            if written is None or written <= 0:
                raise OSError("diagnostic write made no progress")
            view = view[written:]
    finally:
        if raw is not None:
            raw.close()  # closefd=False: the caller retains ownership.



def _standard_write(text, destination=None):
    _transport(text, destination)


class Console:
    def __init__(self, write, max_messages=128, max_bytes=65536,
                 max_line_bytes=4096):
        if any(not isinstance(limit, int) or limit <= 0
               for limit in (max_messages, max_bytes, max_line_bytes)):
            raise ValueError("console limits must be positive integers")
        self._write = write
        self._max_messages = max_messages
        self._max_bytes = max_bytes
        self._line_bytes = min(max_line_bytes, max_bytes)
        self._condition = threading.Condition()
        self._queue = deque()
        self._progress = None
        self._bytes = 0
        self._closed = False
        self._active = False
        self._disabled = [False, False]  # stdout, stderr; fixed-size channel state.
        self._dropped = self._unreported = self._failures = 0
        self._worker = threading.Thread(target=self._run, name="console-writer",
                                        daemon=True)
        self._worker.start()

    def _bounded(self, text):
        text = str(text)
        # Bound the temporary encoding too. Truncation always ends a line.
        data = text[:self._line_bytes + 1].encode("utf-8", "replace")
        if len(data) <= self._line_bytes:
            return data.decode("utf-8")
        marker = ("." * min(3, self._line_bytes - 1) + "\n").encode()
        prefix = data[:self._line_bytes - len(marker)].decode("utf-8", "ignore")
        return prefix + marker.decode()

    def emit(self, text, progress_key=None):
        """Return whether admitted; text is truncated to the UTF-8 byte limit."""
        return self._emit(text, progress_key)

    def _emit(self, text, progress_key=None, destination=None):
        if destination is None and self._write is _standard_write:
            destination = (0, sys.stdout)
        text = self._bounded(text)
        size = len(text.encode("utf-8"))
        record = (text, size, progress_key, destination)
        with self._condition:
            if self._closed:
                return False
            previous = self._progress if progress_key is not None else None
            count = len(self._queue) + (self._progress is not None)
            new_bytes = self._bytes + size - (previous[1] if previous else 0)
            if count + (previous is None) > self._max_messages or new_bytes > self._max_bytes:
                self._dropped += 1
                self._unreported += 1
                return False
            if progress_key is None:
                self._queue.append(record)
            else:
                self._progress = record
            self._bytes = new_bytes
            self._condition.notify_all()
            return True

    def discard_progress(self, key):
        with self._condition:
            if self._progress is not None and self._progress[2] is key:
                self._bytes -= self._progress[1]
                self._progress = None
                self._condition.notify_all()

    @property
    def stats(self):
        """Snapshot; queued counts exclude the record currently in the sink."""
        with self._condition:
            return dict(queued_messages=len(self._queue) + (self._progress is not None),
                        queued_bytes=self._bytes, dropped=self._dropped,
                        failures=self._failures, closed=self._closed)

    def wait_idle(self, timeout=.2):
        """Explicit bounded drain for tests/callers; never used by producers.

        Includes the in-flight write and overload summary. Concurrent producers
        may publish again after this returns; it is not a publication barrier.
        """
        with self._condition:
            return self._condition.wait_for(
                lambda: not self._queue and self._progress is None and not self._active,
                timeout=max(0, timeout))

    def close(self, timeout=.2):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if threading.current_thread() is not self._worker:
            self._worker.join(max(0, timeout))
        return not self._worker.is_alive()

    def _deliver(self, text, destination):
        channel = None
        if self._write is _standard_write and isinstance(destination, tuple):
            channel, destination = destination
            if self._disabled[channel]:
                with self._condition:
                    self._dropped += 1
                return False
        try:
            if self._write is _standard_write:
                self._write(text, destination)
            else:
                self._write(text)
            return True
        except Exception:
            with self._condition:
                self._failures += 1
                if channel is not None:
                    self._disabled[channel] = True
            return False

    def _run(self):
        while True:
            with self._condition:
                while not self._queue and self._progress is None:
                    if self._closed:
                        return
                    self._condition.wait()
                if self._queue:
                    record = self._queue.popleft()
                else:
                    record, self._progress = self._progress, None
                self._bytes -= record[1]
                self._active = True
            # No sink calls, including drop summaries, under the queue lock.
            if self._deliver(record[0], record[3]):
                with self._condition:
                    dropped, self._unreported = self._unreported, 0
                if dropped:
                    summary = self._bounded(f"[console: dropped {dropped} diagnostics]\n")
                    if not self._deliver(summary, record[3]):
                        with self._condition:
                            self._unreported += dropped
            del record  # Do not retain keys or custom streams while idle.
            with self._condition:
                self._active = False
                self._condition.notify_all()


_console = None
_console_lock = threading.Lock()
_shutdown = False


def _get_console():
    global _console
    with _console_lock:
        if _console is None and not _shutdown:
            _console = Console(_standard_write)
        return _console


def console_print(*args, sep=' ', end='\n', file=None, flush=False):
    """Asynchronous diagnostic print; flush is accepted but never waits.

    Explicit custom streams are retained only while their records are pending
    or in flight, with the target captured at publication time. Streams with
    FDs are borrowed, never duplicated or written through Python buffering.
    Windows uses an unbuffered Unicode-aware console/file transport; POSIX
    writes the borrowed FD directly. FD transports use platform-default newline
    conversion (LF to CRLF on Windows); custom text-stream newline modes cannot
    be inferred. Custom/StringIO fallbacks write and flush on the worker;
    blocking custom buffered IO does NOT have the FD transport's
    interpreter-exit guarantee. Callers must keep streams open and avoid
    rebinding their FDs until wait_idle() succeeds, including after a timed-out
    close().
    """
    sep = ' ' if sep is None else sep
    end = '\n' if end is None else end
    if not isinstance(sep, str) or not isinstance(end, str):
        raise TypeError("sep and end must be strings or None")
    console = _get_console()
    if console is not None:
        destination = ((0, sys.stdout) if file is None or file is sys.stdout else
                       (1, sys.stderr) if file is sys.stderr else file)
        console._emit(sep.join(map(str, args)) + end, destination=destination)


def progress(text, key):
    """Publish a progress line with a stable, non-None opaque identity token.

    Use object() stored for the request lifetime, not a service/status object.
    """
    if key is None:
        raise ValueError("progress key must not be None")
    console = _get_console()
    if console is not None:
        text = str(text)
        console.emit(text if text.endswith("\n") else text + "\n", progress_key=key)


def invalidate_progress(key):
    if _console is not None:
        _console.discard_progress(key)


def wait_idle(timeout=.2):
    """Explicitly drain an existing singleton without creating one."""
    console = _console
    return console.wait_idle(timeout) if console is not None else True


def shutdown(timeout=.2):
    global _shutdown
    with _console_lock:
        _shutdown = True
        console = _console
    return console.close(timeout) if console is not None else True


atexit.register(shutdown)


def _after_fork():
    # Threads and their locks do not survive fork; the child owns a new writer.
    global _console, _console_lock, _shutdown
    _console = None
    _console_lock = threading.Lock()
    _shutdown = False


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)
