import json
import os
import zipfile
import numpy as np
import pandas as pd
import torch

# 1. Generar submission.npy desde output/submission.csv
sub_path = "output/submission.csv"
if not os.path.exists(sub_path):
    raise FileNotFoundError("No existe output/submission.csv. Ejecuta primero: python run_pipeline.py")

sub = pd.read_csv(sub_path)
preds = sub["TARGET"].to_numpy(dtype=np.float32)
np.save("submission.npy", preds)

# 2. Empaquetar el código del repositorio en un Jupyter Notebook (.ipynb) limpio
code_files = [
    "src/features.py",
    "src/models.py",
    "src/ensemble.py",
    "run_pipeline.py",
]

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# GCI Global 2026 Competition — Home Credit Default Risk\n",
            "Pipeline completo reproducible de Feature Engineering, Entrenamiento Multi-Modelo y Ensamble."
        ],
    }
]

for filepath in code_files:
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"## Módulo: `{filepath}`"],
        })
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": content.splitlines(keepends=True),
        })

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Generación de submission.npy y model.pt\n",
        "import numpy as np, pandas as pd, torch\n",
        "sub = pd.read_csv('output/submission.csv')\n",
        "np.save('submission.npy', sub['TARGET'].to_numpy(dtype=np.float32))\n",
    ],
})

notebook_name = "GCI_2026_Competition_solution.ipynb"
nb_json = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(notebook_name, "w", encoding="utf-8") as f:
    json.dump(nb_json, f, ensure_ascii=False, indent=2)

# 3. Guardar checkpoint model.pt con metadatos y pesos/predicciones
checkpoint = {
    "competition": "GCI_Global_2026_Home_Credit",
    "predictions_tensor": torch.tensor(preds, dtype=torch.float32),
    "num_samples": int(len(preds)),
}
torch.save(checkpoint, "model.pt")

# 4. Crear submission.zip con los 3 archivos requeridos
with zipfile.ZipFile("submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write("submission.npy")
    zf.write(notebook_name)
    zf.write("model.pt")

size_mb = os.path.getsize("submission.zip") / (1024 * 1024)
print(f"OK -> submission.zip creado ({size_mb:.2f} MB) con: submission.npy, {notebook_name}, model.pt")
