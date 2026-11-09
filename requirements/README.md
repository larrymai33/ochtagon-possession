# Portable CPU runtime locks

These separate locks preserve the numerical-library versions used by the frozen model while selecting official CPU/platform wheels. The verified Windows lock remains unchanged at `../package/repro/requirements-windows-cp311.lock`.

| Lock | Interpreter | Platform minimum | Torch |
| --- | --- | --- | --- |
| `linux-cpu.lock` | CPython 3.11.x | Linux x86_64, glibc 2.28 or later | 2.11.0+cpu |
| `macos-arm64.lock` | CPython 3.11.x | macOS 12 or later, native Apple Silicon/arm64 Python | 2.11.0 |

Both profiles retain NumPy 1.26.4, pandas 1.5.2, scikit-learn 1.3.2, SciPy 1.10.1 and the same eighteen non-Torch distribution versions as the Windows lock. The Mac Torch wheel is tagged macOS 11; SciPy and scikit-learn raise this complete profile's minimum to macOS 12. Linux requires glibc; these wheels do not support Alpine/musl. An Intel Python process under Rosetta is not the Apple Silicon profile.

Each lock contains nineteen exact wheel URLs and SHA-256 hashes, requires binary wheels, and disables package-index fallback. `SOURCES.json` records official release/index origins, fetched wheel metadata, active dependency checks and platform requirements. Metadata was checked on 9 November 2026; wheel binaries were not downloaded for these checks. Installation, model loading and full inference on these profiles remain unverified until separate runtime evidence is recorded.

PyTorch stopped official Intel macOS wheels after the 2.2 series; this package has no native Intel Mac profile. Do not silently substitute an older Torch version. Use a supported Linux environment locally if needed; competition data must remain local and outside the repository.

Matching library versions and repeated inference on one machine do not establish equality across platforms. The immutable CSV writer emits platform-native line endings: Windows CRLF and Linux/macOS LF. Keep raw output hashes distinct from any separately defined canonical row comparison. The frozen source and model must not be changed to force a match.

Official references: [PyTorch installation versions](https://pytorch.org/get-started/previous-versions/), [CPU wheel index](https://download.pytorch.org/whl/cpu/torch/), [Intel Mac deprecation](https://dev-discuss.pytorch.org/t/pytorch-macos-x86-builds-deprecation-starting-january-2024/1690), [NumPy 1.26.4](https://pypi.org/project/numpy/1.26.4/), [pandas 1.5.2](https://pypi.org/project/pandas/1.5.2/), [scikit-learn 1.3.2](https://pypi.org/project/scikit-learn/1.3.2/), and [SciPy 1.10.1](https://pypi.org/project/scipy/1.10.1/).
