# Quick Start Guide

## Option 1: Jupyter Notebook (Recommended for Learning)

1. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

2. Open the notebook:

   ```powershell
   jupyter notebook notebooks/burnout_model.ipynb
   ```

3. Run all cells to train and test the model

## Option 2: Web Application (Production Ready)

1. Train the model first (choose one):

   ```powershell
   # Option A: Use Jupyter notebook (see above)
   # Option B: Use Python script
   python src/model.py
   ```

2. Start the API server:

   ```powershell
   cd src
   python api.py
   ```

3. Open http://localhost:8000 in your browser

## What You Get

- ✅ Binary classification neural network
- ✅ Manual backpropagation implementation
- ✅ Interactive web interface
- ✅ REST API for predictions
- ✅ Complete documentation

## Troubleshooting

**ModuleNotFoundError**: Run `pip install -r requirements.txt`

**Port already in use**: Change port in `src/api.py` (line at the bottom)

**Model not found**: Train the model first using notebook or `python src/model.py`
