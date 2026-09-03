# gate2code reproducibility package

This repository contains the public computational companion to *Native CCZ codes from projective
geometry: a 48-qubit code and a length lower bound*. The accompanying
[supplement](supplement.pdf) gives human-readable certificates, while
[the reproduction guide](docs/REPRODUCING.md) maps each computational claim to its program,
retained output, expected result, and approximate running time.

The public package contains the exact inputs and outputs used by the paper. It excludes unrelated
development files and private repository history.

## Quick start

Create a Python environment and install the package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run the short checks for Supplementary Notes A and B:

```bash
PYTHONPATH=. python scripts/verify_jacinto_d3.py
PYTHONPATH=. python scripts/q48_min_logical_basis.py
PYTHONPATH=. python scripts/verify_tpattern_paper_form.py
PYTHONPATH=. python scripts/verify_native_ladder.py
```

The longer finite searches supporting Supplementary Note C have retained machine-readable outputs.
Consult the reproduction guide before repeating those calculations.

## License

The code is available under the [MIT License](LICENSE).
