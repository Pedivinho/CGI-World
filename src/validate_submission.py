#!/usr/bin/env python
import sys
import pandas as pd
from pathlib import Path

def validate():
    ROOT = Path(__file__).resolve().parents[1]
    sub_path = ROOT / 'output' / 'submission.csv'
    ref_path = ROOT / 'Competition' / 'input' / 'sample_submission.csv'
    
    print(f'Validando {sub_path}...')
    if not sub_path.exists():
        print('ERROR: No existe el archivo de submission.')
        return 1
    
    try:
        sub = pd.read_csv(sub_path)
        ref = pd.read_csv(ref_path)
    except Exception as e:
        print(f'ERROR leyendo archivos: {e}')
        return 1
    
    # 1. Filas
    if len(sub) != 61500:
        print(f'ERROR: Filas incorrectas ({len(sub)} != 61500)')
        return 1
    
    # 2. Columnas
    expected_cols = ['SK_ID_CURR', 'TARGET']
    if list(sub.columns) != expected_cols:
        print(f'ERROR: Columnas incorrectas ({list(sub.columns)} != {expected_cols})')
        return 1
    
    # 3. SK_ID_CURR (Orden y Valores)
    if not (sub['SK_ID_CURR'].values == ref['SK_ID_CURR'].values).all():
        print('ERROR: SK_ID_CURR no coincide con el orden/valores de sample_submission.')
        return 1
    
    # 4. NaNs / Infs
    if sub.isna().any().any():
        print('ERROR: Hay valores NaN en la submission.')
        return 1
    if (sub[['TARGET']] == float('inf')).any().any():
        print('ERROR: Hay valores Inf en la submission.')
        return 1
        
    # 5. Rango TARGET
    t_min, t_max = sub['TARGET'].min(), sub['TARGET'].max()
    if t_min < 0.0 or t_max > 1.0:
        print(f'ERROR: TARGET fuera de rango [0, 1] (min={t_min}, max={t_max})')
        return 1
        
    print('VALIDACIÓN EXITOSA: Submission lista para Omnicampus.')
    return 0

if __name__ == '__main__':
    sys.exit(validate())
