"""serve/test_first_chunk.py - #2391: no data chunk before the prompt is read; the role chunk first, exactly once.

    python -m unittest serve.test_first_chunk -v
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from serve.frontend import Event, ToolCall  # noqa: E402
from serve.server import openai_chunks, openai_collect  # noqa: E402

DONE = ("done", {"finish": "stop", "completion_tokens": 0, "prompt_tokens": 3})
ROLE = {"role": "assistant", "content": ""}


class FakeSvc:
    """Service.run's shape: PP pings while the prompt is read, then the events, then the finish."""

    def __init__(self, items, progress=None):
        self.items, self.read = items, False
        self.engine = None
        self.progress = progress

    def model_for(self, req):
        return "m"

    def run(self, ids, thinking, tools, max_new, req, cancel, force=None):
        for item in self.items:
            if item == "prompt-read":
                self.read = True
                continue
            yield item


def stream(items, req=None):
    svc = FakeSvc(items)
    out = []
    for c in openai_chunks(svc, req or {}, [1, 2, 3], False, None, 16, None):
        out.append((svc.read, c))
    return out


class RoleChunkAfterPrefill(unittest.TestCase):
    def check(self, out):
        data = [(read, c) for read, c in out if c is not None]
        self.assertTrue(data)
        self.assertTrue(all(read for read, _ in data), "a data chunk went out before the prompt was read")
        self.assertEqual(data[0][1]["choices"][0]["delta"], ROLE)
        roles = [c for _, c in data if "role" in c["choices"][0]["delta"]]
        self.assertEqual(len(roles), 1)
        return [c for _, c in data]

    def test_content(self):
        chunks = self.check(stream([("ping", None), ("ping", None), "prompt-read",
                                    ("event", Event("content", "hi")), DONE]))
        self.assertEqual(chunks[1]["choices"][0]["delta"], {"content": "hi"})
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")

    def test_pings_stay_comments(self):
        out = stream([("ping", None), ("ping", None), "prompt-read", ("event", Event("content", "x")), DONE])
        self.assertEqual([c for _, c in out[:2]], [None, None])

    def test_empty_events_do_not_open_stream(self):
        chunks = self.check(stream([("event", Event("content", "")),
                                    ("event", Event("reasoning", "")),
                                    ("ping", None), "prompt-read",
                                    ("event", Event("content", "hi")), DONE]))
        self.assertEqual(chunks[1]["choices"][0]["delta"], {"content": "hi"})

    def test_empty_answer(self):
        chunks = self.check(stream([("ping", None), "prompt-read", DONE]))
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[1]["choices"][0]["finish_reason"], "stop")

    def test_streamed_tool_call(self):
        call = ToolCall("f", {"a": 1})
        chunks = self.check(stream([("ping", None), "prompt-read", ("event", Event("tool_start", call=call)),
                                    ("event", Event("tool_args", '{"a": 1}', call=call)),
                                    ("event", Event("tool_call", call=call)), DONE]))
        self.assertEqual(chunks[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"], "f")
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "tool_calls")

    def test_whole_tool_call(self):
        chunks = self.check(stream([("ping", None), "prompt-read",
                                    ("event", Event("tool_call", call=ToolCall("f", {}))), DONE]))
        self.assertIn("tool_calls", chunks[1]["choices"][0]["delta"])

    def test_error_before_first_token(self):
        def boom():
            yield ("ping", None)
            raise ValueError("engine ERR")
        svc = FakeSvc([])
        svc.run = lambda *a, **k: boom()
        got = []
        with self.assertRaises(ValueError):
            for c in openai_chunks(svc, {}, [1], False, None, 16, None):
                got.append(c)
        self.assertEqual(got, [None])                 # no role chunk: the caller sends the error as the first data

    def test_collect_unchanged(self):
        svc = FakeSvc([("ping", None), ("event", Event("content", "hi")), DONE])
        res = openai_collect(openai_chunks(svc, {}, [1, 2, 3], False, None, 16, None))
        self.assertEqual(res["choices"][0]["message"]["content"], "hi")


if __name__ == "__main__":
    unittest.main()
