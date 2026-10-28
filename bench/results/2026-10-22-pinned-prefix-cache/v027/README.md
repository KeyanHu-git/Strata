# v0.2.7 cache validation

Base `5e8a4cca`; tested patch `da63fde09a7f83692cf9de5c05bf4001a46d7bed`.
Windows 11, Ryzen 5 7600X, RX 7900 XTX 24 GiB, 32 GiB RAM, Coder IQ1_M.

| Result | v0.2.7 | Patch | Patch, async saves enabled |
|---|---:|---:|---:|
| Requests | 15 | 15 | 15 |
| Parking refusals | 8 | 0 | 0 |
| Restores | 0 | 6 | 6 |
| Input tokens read on six revisits | 1778 | 90 | 90 |
| Output sequences differing from base | — | 0 | 0 |

The three arms reuse [requests.txt](../requests.txt) and [fixtures.json](../fixtures.json).
Each starts a fresh engine, uses greedy generation with seed 2255, a 196-token pin,
three slots and a 4 GiB conversation-cache budget. Expert placement is fixed:
`--expert-cache 4416 --pcie-frac 0 --adapt-swaps 0 --resident-budget-gib 8`.
`STRATA_SNAPSHOT_VERIFY=1` is enabled. The third arm additionally enables
`STRATA_PARK_ASYNC=1 STRATA_CKPT_ASYNC=1`. This validates output/reuse under those
settings; it does not measure copy overlap or establish a throughput gain.
The three-arm build/test driver took 320.91 seconds.

[model-results.json](model-results.json) contains every request, output IDs, cache
messages, exact non-path settings and binary hashes. Workstation paths are replaced
with placeholders. Use the parent [replay.py](../replay.py) with the recorded
settings and each executable to repeat the requests.

## Pending-copy ownership

v0.2.7 stores copy-completion tokens after the snapshot buffers so destruction joins
copies first. Erasing a middle deque entry instead move-assigns its neighbour:
memberwise assignment can release the victim's buffers before its copy tokens.
The patch moves the victim into a local owner before shifting the deque. Its normal
destructor then joins pending copies before releasing storage.

The cache regression constructs a redundant middle holder with pending conversation,
live-state and layer-K/V tokens. Their deleters inspect the corresponding buffers;
the original erase path reports an ASan heap-use-after-free, while the fixed path
joins all three tokens with live storage. This is a CPU ownership counterexample,
not a claimed GPU crash. The upstream fixture for a 22-token parked image also now
contains the 22nd token, allowing the whole cache suite to run under ASan.

## Checks

The parent README contains the exact sanitizer commands. On this revision:
45,222 cache checks and 161 session-save checks pass with ASan/UBSan on macOS.
[Source hashes](source-hashes.json), [cache output](cache.log), [session output](session.log).
Full CUDA (sm_120), HIP (gfx1100), SYCL and Metal engine builds passed.
CUDA ctest passed both conversation-cache and session-save checks; the HIP test
executables passed as well. [Build receipts](build-results.json) separate builds
from model execution. The first CUDA configure failed because the transferred
archive omitted bench/micro/fusions_parity.cu; restoring the complete bench
directory and rerunning the same build script passed.
Model execution above is HIP only; CUDA/SYCL/Metal model execution was not run.
