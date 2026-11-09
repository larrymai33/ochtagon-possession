# Portable CPU runtime locks

These separate locks preserve the numerical-library versions used by the frozen model while selecting official CPU/platform wheels. The verified Windows lock remains unchanged at `../package/repro/requirements-windows-cp311.lock`.

| Lock | Interpreter | Platform minimum | Torch |
| --- | --- | --- | --- |
| `linux-cpu.lock` | CPython 3.11.x | Linux x86_64, glibc 2.28 or later | 2.11.0+cpu |
| `macos-arm64.lock` | CPython 3.11.x | macOS 12 or later, native Apple Silicon/arm64 Python | 2.11.0 |

Both profiles retain NumPy 1.26.4, pandas 1.5.2, scikit-learn 1.3.2, SciPy 1.10.1 and the same eighteen non-Torch distribution versions as the Windows lock. The Mac Torch wheel is tagged macOS 11; SciPy and scikit-learn raise this complete profile's minimum to macOS 12. Linux requires glibc; these wheels do not support Alpine/musl. An Intel Python process under Rosetta is not the Apple Silicon profile.

Each lock contains nineteen exact wheel URLs and SHA-256 hashes, requires binary wheels, and disables package-index fallback. `SOURCES.json` records the metadata checks made on 9 November 2026: official release/index origins, wheel metadata, active dependency checks and platform requirements. Its scope describes that original metadata stage. Later installation and inference checks are recorded separately in [verification/PLATFORMS.md](../verification/PLATFORMS.md).

The Linux lock passed fresh installation, model inference, official CSV validation and exact final reference-hash matching on local Ubuntu WSL: x86_64, glibc 2.43, CPython 3.11.18 and Torch 2.11.0+cpu. The Windows Python workflow also passed from the same clean code commit, `7f27085bfc9024391b22deda827b27ef1fc39f42`, on the same physical host. Windows passed 20 release tests; Linux passed 16 with four PowerShell-specific tests skipped. This establishes the tested environments, not every compatible machine or distribution.

The macOS download check retrieved all nineteen wheels, totaling 154,559,425 bytes, and verified their hashes, sizes and embedded metadata. Installation, imports, model loading and inference on macOS hardware remain untested.

PyTorch stopped official Intel macOS wheels after the 2.2 series; this package has no native Intel Mac profile. Do not silently substitute an older Torch version. Use a supported Linux environment locally if needed; competition data must remain local and outside the repository.

Matching library versions alone does not establish equality across platforms. The immutable CSV writer emits platform-native line endings: Windows CRLF and Linux/macOS LF. `scripts/reproduce.py` preserves that original CSV and receipt under the run folder's `native-output/`, then serializes only the newlines as CRLF in the final `output/ochtagon.csv`. Frames and labels remain unchanged. The final file must match the same reference SHA-256 as Windows; otherwise the run fails. Native and final hashes remain separate, and the frozen model and source remain unchanged.

Official references: [PyTorch installation versions](https://pytorch.org/get-started/previous-versions/), [CPU wheel index](https://download.pytorch.org/whl/cpu/torch/), [Intel Mac deprecation](https://dev-discuss.pytorch.org/t/pytorch-macos-x86-builds-deprecation-starting-january-2024/1690), [NumPy 1.26.4](https://pypi.org/project/numpy/1.26.4/), [pandas 1.5.2](https://pypi.org/project/pandas/1.5.2/), [scikit-learn 1.3.2](https://pypi.org/project/scikit-learn/1.3.2/), and [SciPy 1.10.1](https://pypi.org/project/scipy/1.10.1/).
