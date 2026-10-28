# Console diagnostics

Serving publishes console diagnostics to one process-owned writer. A paused
terminal or a full output pipe does not make request execution wait for a
progress line, completion message or error diagnostic. Engine protocol traffic,
HTTP responses and model tokens use their existing channels. The human-readable
`ready:` banner is best-effort; automation should use the ready file, structured
lifecycle events or HTTP readiness checks.

The writer has at most 128 pending records and 64 KiB of pending UTF-8 text,
plus one record being written. Each record is limited to 4 KiB. Long messages
end with a truncation marker. These limits apply to queued diagnostics, not to
request data or engine log files.

Progress keeps only the latest pending sample. Ending a request removes its
pending sample without removing another request's sample. A line already being
written can appear after the request ends. Under sustained output backpressure,
new diagnostics that do not fit are dropped; after output resumes, the writer
reports the dropped count. A failed standard output channel is disabled for the
rest of the process. Console output is therefore best-effort, not an audit log.

For stdout/stderr with file descriptors, the writer borrows the descriptor
and uses unbuffered writes. Windows console handles retain Python's Unicode
console conversion. The writer does not hold Python's buffered-stream lock
during a blocked write. Callers must keep the stream and descriptor open until
their diagnostics have drained. FD-backed output uses platform-default newline
conversion, including LF-to-CRLF on Windows; custom stream newline settings are
not inferred. Shutdown waits at most 0.2 seconds for diagnostics; pending lines can be
lost on exit. Embedded applications supplying custom buffered streams need to
keep those streams open until drained; their shutdown behavior depends on the
custom stream.

`before_load` hooks keep their existing exit-code and timeout behavior. Their
stdout/stderr go to a temporary file so they do not inherit a blocked terminal.
Only the last 3072 bytes are submitted after the hook finishes or times out.
The spool's disk size depends on the hook's output. Capture failures discard
that output without changing the hook's result. Hooks now see a file rather than
a terminal on stdout/stderr.

The normal queue and progress API never waits for actual output. The explicit
`wait_idle` and `close` methods are for bounded draining and tests, not the
request path. Configured third-party logging handlers and commands that open
the console themselves are separate output paths.
