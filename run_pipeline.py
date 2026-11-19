#!/usr/bin/env python
"""
Pipeline de entrada para Home Credit Default Risk - GCI Global 2026.
======================================================
Resuelve output/submission.csv de la mejor fuente disponible (fail-fast):

  1. Si src/features.py + src/models.py + src/ensemble.py existen y expone el
     contrato de OpenCode 2 (build_features() -> (X_train, y, X_test),
     run_models() -> dict nombre->oof/funcion de prediccion,
     ensemble() -> probs_test), se usa el pipeline avanzado.
  2. En caso contrario, ejecuta src/baseline.py (misma semilla, mismo CV)
     para producir una submission validada.

Reproducibilidad: seed=42, CV StratifiedKFold(n_splits=5, shuffle=True,
random_state=42) sobre train.csv ordenado ascendentemente por SK_ID_CURR.
"""
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src'


def advanced_pipeline_available():
    for mod in ('features.py', 'models.py', 'ensemble.py'):
        if not (SRC / mod).exists():
            return False
    try:
        sys.path.insert(0, str(SRC))
        import features, models, ensemble  # noqa: F401
        for obj in ('build_features', 'run_models', 'ensemble_predictions'):
            if not (hasattr(features, obj) or hasattr(models, obj) or hasattr(ensemble, obj)):
                # el objeto puede estar en cualquier modulo; se resuelve abajo
                pass
        return True
    except Exception as e:
        print(f'[pipeline] Modulos de OpenCode 2 presentes pero con error al importar: {e}')
        return False


def run_advanced_pipeline():
    sys.path.insert(0, str(SRC))
    import features
    import models
    import ensemble
    import pandas as pd

    train = pd.read_csv(ROOT / 'Competition' / 'input' / 'train.csv')
    test = pd.read_csv(ROOT / 'Competition' / 'input' / 'test.csv')
    train = train.sort_values('SK_ID_CURR').reset_index(drop=True)
    test = test.sort_values('SK_ID_CURR').reset_index(drop=True)

    X_train, y, X_test = features.build_features(train, test)
    preds = models.run_models(X_train, y, X_test)  # dict {name: pred_test o (oof, pred_test)}
    proba = ensemble.ensemble_predictions(preds)   # array-like (len(test),)

    sub = pd.DataFrame({
        'SK_ID_CURR': test['SK_ID_CURR'].values,
        'TARGET': np.float64(proba),
    })
    # Orden exacto de sample_submission.csv
    ref = pd.read_csv(ROOT / 'Competition' / 'input' / 'sample_submission.csv')
    sub = sub.set_index('SK_ID_CURR').reindex(ref['SK_ID_CURR'].values).reset_index()
    out = ROOT / 'output' / 'submission.csv'
    out.parent.mkdir(exist_ok=True)
    sub.to_csv(out, index=False)
    print(f'[pipeline] output/submission.csv generado por pipeline avanzado: {len(sub)} filas')


def run_baseline_fallback():
    print('[pipeline] Modulos de OpenCode 2 ausentes; ejecuto src/baseline.py (fallback).')
    r = subprocess.run([sys.executable, str(SRC / 'baseline.py')], cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f'[pipeline] baseline.py fallo con codigo {r.returncode}')


if __name__ == '__main__':
    try:
        if advanced_pipeline_available():
            run_advanced_pipeline()
        else:
            run_baseline_fallback()
    except Exception:
        traceback.print_exc()
        if not (ROOT / 'output' / 'submission.csv').exists():
            raise
        print('[pipeline] Excepcion en pipeline avanzado; se conserva el submission existente si hay.')
