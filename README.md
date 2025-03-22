# stretch-model-cosmology
# Stretch-Driven Cosmological Model

This repository contains the data, code, and analysis supporting the manuscript:

**"Dark Energy as Cloth Stretching: A Stretch-Driven Anti-Gravity Model for Cosmic Expansion and Structure Formation"** (submitted to ApJ, March 2025)

## 📁 Contents

- `figure_data/`: CSV and `.npy` files used to generate Figures 1–8
- `growth_factor_plot.py`: Code used for computing and plotting the growth factor (Figure 8)
- `README.md`: Overview of the project and dataset

## 📊 Data Behind the Figures

All datasets used in the paper are available in the `figure_data/` folder.

## 🔗 Links

- 📄 [Manuscript PDF (arXiv link when available)](https://arxiv.org/)
- 🔬 [JWST Dataset reference: GO 1180, JADES](https://archive.stsci.edu/)

## 🧪 Reproducibility

- Python 3.8+
- NumPy, SciPy, matplotlib

To reproduce Figure 8:
```bash
python growth_factor_plot.py
