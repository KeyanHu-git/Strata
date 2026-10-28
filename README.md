# Move-only conversation snapshot ownership

Source: ce957e1457a4d82329fe40ac8dfb279fb21a008d. Base: v0.2.7 5e8a4cca.
Ryzen 5 7600X, RX 7900 XTX, Windows 11, Coder IQ1_M.
Three arms, 15 requests each; greedy generation, three cache slots, 4 GiB cache budget,
fixed expert placement and snapshot verification. Model paths are placeholders.

| Result | Upstream | Fixed | Fixed with async flags |
|---|---:|---:|---:|
| Parking refusals | 8 | 0 | 0 |
| Restores | 0 | 6 | 6 |
| Revisit input tokens processed | 1778 | 90 | 90 |
| Output sequences differing from upstream | — | 0 | 0 |

The successful retry driver took 200.39 seconds, including incremental HIP build,
four CPU tests, four snapshot configurations and the three model arms.
Initial build time is separate. Full token sequences and binary hashes are retained.

HIP explicitly disables both park_async_on and ckpt_async_on. The async flag arm
checks the fallback configuration, not CUDA background DMA or copy overlap.
The snapshot variants likewise run the HIP fallback. Pending-token destruction is
covered separately by sanitizer regressions. CUDA asynchronous device execution
remains untested in this batch.

Eight compiler probes reject copy construction and assignment of ConversationKv,
SavedConversation, ConversationKvReuse and ConversationCache. CPU sanitizer checks:
45,257 cache, 198 file-format, 161 reclamation. Regression coverage includes old/new
pending owners, self-move, both deque shift directions, nested stages, donor
protection and matching admission/eviction accounting.

Initial CUDA/SYCL builds rejected a leftover copy assignment in malformed-payload
test fixtures. Both fixtures now capture independent images. No production source
changed between 6ebe05c2 and ce957e14.

Reproduction: fill the paths in settings.example.json, then run
`python replay.py --engine /path/to/strata --settings settings.json --output run`.
Use model-results.json for exact arm settings. This experiment establishes cache
reuse and matching outputs, not a throughput claim.

Full CUDA, HIP, Metal and SYCL engines build. CUDA CPU ctest passes 5/5.
CUDA and SYCL snapshot device-test targets compile. Windows test engines exited
and port8080 was verified idle after the run.
