# ML-PWS: Machine Learning Implementation of Path Weight Sampling


[![Static Badge](https://img.shields.io/badge/arXiv-2508.16509-black)](https://arxiv.org/abs/2508.16509)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/manuel-rhdt/ml-pws/HEAD)

ML-PWS is a Python package that implements Path Weight Sampling (PWS) models for time series analysis. It provides tools for computing mutual information in sequential data using autoregressive models and Monte Carlo Sampling, built with JAX, Flax, and PyTorch.

## Installation

### Prerequisites

- Python 3.11 or higher

### Installing with conda

```bash
conda env create -f environment.yml
```

### Installing with uv

1. First, install uv if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone the repository:
```bash
git clone https://github.com/manuel-rhdt/ml-pws.git
cd ml-pws
```

3. Create and activate a new virtual environment with uv:
```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

4. Install the package and its dependencies:
```bash
uv pip install -e .
```

## Usage

The package provides several scripts to recreate the figures from the paper:

1. Regenerate figure datasets:
```bash
make data
```

2. Scripts to generate figures are in the `scripts` directory. You can create Figure 1 from the publication using
```bash
python scripts/figure1.py
```
which will save the figure in `reports/figures`.
![Figure 1](reports/figures/ml_pws_fig_1.png)

## Project Structure

- `ml_pws/`: Main package directory
  - `data/`: Dataset generation
  - `models/`: Model implementations
- `scripts/`: Analysis and utility scripts
- `notebooks/`: Jupyter notebooks for examples and analysis
- `experiments/`: Experiment results
- `data/`: Example datasets and results

## Dependencies

- clu >= 0.0.12
- flax >= 0.11.2
- jax >= 0.7.2
- lightning >= 2.5.5
- matplotlib >= 3.10.6
- numpy >= 2.0.2
- polars >= 1.33.1
- scipy >= 1.16.2
- tensorflow >= 2.20.0
- torch >= 2.8.0
- tqdm >= 4.67.1

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Manuel Reinhardt
