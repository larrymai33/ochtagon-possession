# Portable runtime verification

Checks recorded on 9 November 2026 for implementation commit
`7f27085bfc9024391b22deda827b27ef1fc39f42`. Setup used the matching committed
working files; inference used a separate clean local clone of that commit.
The frozen package, model and original Windows lock were unchanged.

## Results

| Profile | Runtime | Completed verification | Full CSV reproduction |
| --- | --- | --- | --- |
| Windows AMD64 | CPython 3.11.0; Torch 2.11.0+cu128; CPU inference | Fresh setup launched from CMD, dependency checks, 20 tests passed | Passed from the clean clone; exact reference bytes |
| Linux x86_64, local Ubuntu under WSL | CPython 3.11.18; glibc 2.43; Torch 2.11.0+cpu | Fresh setup and dependency checks passed; 16 tests passed, four PowerShell-specific tests skipped | Passed from the clean clone; exact reference bytes after CRLF serialization |
| macOS arm64 | CPython 3.11 wheel target; macOS 12 or later profile | All 19 wheels downloaded and verified against the lock and official metadata | Not executed; no Mac installation, imports, model loading or inference claimed |

Both final CSVs contain **199,319 rows**, covering frames 0 through 199318,
and are **1,882,093 bytes**. Both complete runs exited successfully, passed the
packaged official format validator and matched the reference SHA-256:

```text
35d905533b696bee3f29da3e6c9f24ebab76780982e9f2325b6e67784dfea303
```

The reference identity is in [reference.json](reference.json). Aggregate
platform receipts are recorded separately in [platforms.json](platforms.json).
The Windows and Linux results establish exact final CSV reproduction for
these tested environments, including their distinct native line endings.

## Native output and final serialization

The immutable inference code writes platform-native line endings and repeats
prediction to require identical labels within the run. The portable wrapper
preserves that CSV and its original receipt under `native-output/`.

After official format validation, the wrapper creates a separate final CSV
under `output/`, converting only line endings to CRLF. It preserves every
other byte and checks the final SHA-256 against the Windows reference. It
then repeats the official format and reference check on the final file.
Raw native and final output hashes are recorded separately.

If any prediction differs from the reference, newline conversion cannot hide
that difference: verification fails and no successful reproduction receipt is
written. Existing outputs are never overwritten. A preserved file from a
failed run is not a verified submission.

## Mac wheel evidence

The Mac download check targeted `macosx_12_0_arm64`, CPython 3.11, ABI `cp311`,
using pip 22.3 on Windows. All **19 wheels, totaling 154,559,425 bytes**, matched
their recorded sizes and SHA-256 values. Their embedded dependency metadata
also matched the official metadata fetched when the lock was prepared.

Lock SHA-256:
`63cf427b1809fe0f2dc0538aa38ba3ae0c70ec11316d76337df4dcf9d79888f2`.
External download-validation receipt SHA-256:
`08c7b67e6ce2b5aafc5ef2d870500e0aa8da025fc6f940ff06fd8727914a2ef8`.

This establishes wheel availability and binary integrity, not execution on
Apple hardware. [requirements/SOURCES.json](../requirements/SOURCES.json)
preserves the earlier metadata-only snapshot; this page records the later
download check without rewriting that snapshot or either lock.

## Scope

Windows and WSL checks use the **same physical development host**. WSL supplies
a Linux runtime; it is not evidence from a second machine, different hardware,
or every supported Linux distribution. Mac execution remains unverified.

The [original PowerShell reproduction record](REPRODUCTION.md) is preserved.
This page adds evidence for the Python setup and reproduction commands; it
does not replace the original Windows result.

Setup needs no match data. Inference checks use only the separately supplied
automatic tactical tracking and metadata. No target ground truth, video,
training, parameter selection or accuracy scoring is part of these checks.
No supplied match data is included in these records or uploaded by the
reproduction workflow. Format validation and exact output reproduction do
not establish predictive accuracy, competition eligibility or advancement.
