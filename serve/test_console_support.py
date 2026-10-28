"""Synchronous diagnostic recording for formatting/state unit tests.

Output isolation and writer behavior use the real Console in dedicated tests.
"""
import contextlib
import sys
from unittest import mock
from serve import console


@contextlib.contextmanager
def capture_console(stream):
    class Recorder:
        def _emit(self, text, progress_key=None, destination=None):
            if isinstance(destination, tuple):
                destination = destination[1]
            target = stream if destination is None or destination == 'stdout' else destination
            if target == 'stderr':
                target = sys.stderr
            target.write(text)
            return True

        emit = _emit

        def discard_progress(self, key):
            pass

    with contextlib.redirect_stdout(stream), mock.patch.object(console, '_console', Recorder()):
        yield stream
