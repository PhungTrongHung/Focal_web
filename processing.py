import numpy as np
import pandas as pd
from scipy.stats import skew

SENSORS = [f'{kind}_{axis}' for kind in ('acc', 'gyro', 'mag') for axis in 'xyz']
REQUIRED = ['time_s', *SENSORS]
WINDOW = 200  # For default ~100 Hz data, approximately 2 seconds
STEP = 100


def clean_csv(source):
    df = pd.read_csv(source)
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df]
    if missing:
        raise ValueError('Thiếu các cột: ' + ', '.join(missing))
    df = df.copy()
    for c in REQUIRED:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=REQUIRED).sort_values('time_s').drop_duplicates('time_s')
    if len(df) < WINDOW:
        raise ValueError(f'Cần ít nhất {WINDOW} dòng dữ liệu hợp lệ cho một cửa sổ; hiện có {len(df)} dòng.')
    if not np.isfinite(df[REQUIRED].to_numpy()).all():
        raise ValueError('Dữ liệu chứa giá trị không hữu hạn.')
    return df.reset_index(drop=True)


def windows(df, window=WINDOW, step=STEP):
    for i in range(0, len(df) - window + 1, step):
        yield df.iloc[i:i + window]


def feature_dict(chunk):
    out = {}
    for col in SENSORS:
        x = chunk[col].to_numpy(dtype=float)
        out[f'{col}_mean'] = float(np.mean(x))
        out[f'{col}_std'] = float(np.std(x))
        out[f'{col}_min'] = float(np.min(x))
        out[f'{col}_max'] = float(np.max(x))
        out[f'{col}_rms'] = float(np.sqrt(np.mean(x*x)))
        out[f'{col}_median'] = float(np.median(x))
    for kind in ('acc', 'gyro', 'mag'):
        v = chunk[[f'{kind}_{a}' for a in 'xyz']].to_numpy(dtype=float)
        norm = np.linalg.norm(v, axis=1)
        out[f'{kind}_norm_mean'] = float(norm.mean())
        out[f'{kind}_norm_std'] = float(norm.std())
    return out


def extract_features(df):
    chunks = list(windows(df))
    return pd.DataFrame([feature_dict(c) for c in chunks]), chunks
