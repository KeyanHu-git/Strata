# Snapshot owner replacement on v0.2.7

Tested production revision: `3d502f474a0842f92827f620fd026be23f870657`.
Base: v0.2.7 `5e8a4cca`. Ryzen 5 7600X, RX 7900 XTX 24 GiB, 32 GiB RAM,
Windows 11, Coder IQ1_M. Same 15-request sequence per arm, greedy seed 2255,
196-token pin, three cache slots, 4 GiB cache budget, fixed expert placement.
Snapshot verification enabled. This is ownership/correctness/reuse validation.

| Result | Upstream | Candidate | Candidate, async saves |
|---|---:|---:|---:|
| Parking refusals | 8 | 0 | 0 |
| Restores | 0 | 6 | 6 |
| Input tokens processed on six revisits | 1778 | 90 | 90 |
| Output sequences differing from upstream | — | 0 | 0 |

The build-and-model driver finished in 317.11 seconds. Full request/output records,
binary hashes and sanitized actual settings are in model-results.json. Async arm
adds STRATA_PARK_ASYNC=1 and STRATA_CKPT_ASYNC=1. The settings validate that enabled
configuration; no device-copy overlap or throughput improvement is inferred.

CPU sanitizer results are in cpu-results.json. The direct-assignment counterexample
fails with ASan on the previous implementation and passes on the candidate. New
regressions cover old/new pending owners, self-move, completed-source copy, KV-layer
replacement, both deque shift directions, shifted neighbours and nested stages.

Copying requires completed source writes; moving transfers in-flight ownership.
By-value assignment retires the complete previous owner, so cache erase, snapshot
publication and file-load replacement share the same lifetime guarantee.

To replay, fill the model paths in settings.example.json and run:

```
python replay.py --engine /path/to/strata --settings settings.json --output run
```

The precise per-arm settings are in model-results.json; keep every non-path value.
Build receipts distinguish compilation from HIP model execution. No CUDA, SYCL,
Metal model execution or NPU execution was performed in this batch.
