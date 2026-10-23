import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from torch.nn import functional as F
from focal_model import FocalModel, focal_loss, SENSORS
from torch.utils.data import DataLoader, TensorDataset

COLS = ['time_s', *SENSORS, 'label']


def load_windows(path, win=200, chunksize=50000, max_windows=0):
    """Streaming parser; time resets/gaps/label changes create recording segments.
    For credible evaluation, collect multiple independent sessions for every activity.
    """
    xs, ys, groups = [], [], []
    buffer = []
    group = 0
    prev_time = None
    prev_label = None
    prev_session = None
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        chunk.columns = chunk.columns.str.strip().str.lower()
        missing = [c for c in COLS if c not in chunk]
        if missing:
            raise ValueError(f'Thieu cot CSV: {missing}')
        has_session = 'session_id' in chunk.columns
        chunk = chunk[COLS + (['session_id'] if has_session else [])].copy()
        for c in COLS[:-1]:
            chunk[c] = pd.to_numeric(chunk[c], errors='coerce')
        chunk['label'] = chunk['label'].astype(str).str.strip().str.lower()
        for row in chunk.itertuples(index=False, name=None):
            if has_session:
                t, *values, label, session = row
            else:
                t, *values, label = row
                session = None
            valid = np.isfinite(np.asarray([t, *values], dtype=np.float64)).all() and label not in ('nan', '', 'none')
            if not valid:
                buffer.clear(); prev_time = None; prev_label = None; prev_session = None; group += 1
                continue
            if prev_time is not None and (t <= prev_time or t - prev_time > 0.5 or label != prev_label or session != prev_session):
                buffer.clear(); group += 1
            prev_time, prev_label, prev_session = t, label, session
            buffer.append(values)
            if len(buffer) == win:
                xs.append(np.asarray(buffer, dtype=np.float32))
                ys.append(label); groups.append(f'{session}' if has_session else str(group))
                buffer.clear()  # nonoverlapping windows
                if max_windows and len(xs) >= max_windows:
                    break
        if max_windows and len(xs) >= max_windows:
            break
    if not xs:
        raise ValueError('Khong co cua so du 200 dong lien tuc. File head30 chi de kiem tra dinh dang.')
    return np.stack(xs), np.asarray(ys), np.asarray(groups)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('csv', type=Path)
    p.add_argument('--epochs-pretrain', type=int, default=5)
    p.add_argument('--epochs-classify', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--max-windows', type=int, default=0, help='0=all; set e.g. 15000 for trial')
    p.add_argument('--output', type=Path, default=Path('models/focal.pt'))
    args = p.parse_args()
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    X, y, groups = load_windows(args.csv, max_windows=args.max_windows)
    labels = sorted(str(v) for v in set(y)); print('Windows:', len(X), 'Labels:', {k:int((y==k).sum()) for k in labels}, 'Sessions:',len(set(groups)))
    if len(labels) < 2:
        raise ValueError('Can it nhat 2 nhan. head30 khong du de train.')
    if len(set(groups)) < 3:
        raise ValueError('Khong du 3 doan du lieu doc lap de chia train/val/test. Hay thu them cac phien.')
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, temp = next(gss.split(X, y, groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=43)
    va_rel, te_rel = next(gss2.split(X[temp], y[temp], groups[temp]))
    va, te = temp[va_rel], temp[te_rel]
    for name, idx in [('train',tr),('val',va),('test',te)]:
        print(name, len(idx), sorted(set(y[idx])))
    if not len(va) or not len(te):
        raise ValueError('Khong the tao validation/test tu cac session doc lap.')
    if set(y[tr]) != set(labels):
        raise ValueError('Train khong co du nhan. Can bo sung nhieu session/nguoi thu tham gia.')
    mean = X[tr].mean(axis=(0,1), keepdims=True)
    std = X[tr].std(axis=(0,1), keepdims=True).clip(1e-5)
    X = ((X - mean) / std).astype(np.float32)
    ids = np.array([labels.index(k) for k in y], dtype=np.int64)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:',dev)
    model = FocalModel(len(labels)).to(dev)
    def loader(index, shuffle):
        return DataLoader(TensorDataset(torch.from_numpy(X[index]), torch.from_numpy(ids[index])),
                          batch_size=args.batch_size, shuffle=shuffle)
    train_ld = loader(tr, True); val_ld = loader(va, False); test_ld = loader(te, False)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    for epoch in range(args.epochs_pretrain):
        model.train(); losses=[]
        for xx, _ in train_ld:
            xx=xx.to(dev)
            if xx.shape[0]<2: continue
            loss=focal_loss(model, xx)
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        print(f'Pretrain {epoch+1}: loss={np.mean(losses):.4f}')
    opt = torch.optim.AdamW(model.parameters(),lr=3e-4)
    best=-1; saved=None
    for epoch in range(args.epochs_classify):
        model.train()
        for xx, yy in train_ld:
            logits=model(xx.to(dev)); loss=F.cross_entropy(logits, yy.to(dev))
            opt.zero_grad();loss.backward();opt.step()
        model.eval(); correct=total=0
        with torch.no_grad():
            for xx,yy in val_ld:
                pred=model(xx.to(dev)).argmax(-1).cpu();correct+=int((pred==yy).sum());total+=len(yy)
        accuracy=correct/max(total,1)
        print(f'Finetune {epoch+1}: val accuracy={accuracy:.3f}')
        if accuracy>best:
            best=accuracy; saved={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(saved); model.to(dev).eval()
    preds=[]; truth=[]
    with torch.no_grad():
        for xx,yy in test_ld:
            preds.extend(model(xx.to(dev)).argmax(-1).cpu().tolist());truth.extend(yy.tolist())
    report=classification_report(truth,preds,labels=list(range(len(labels))),target_names=labels,zero_division=0,output_dict=True)
    print(classification_report(truth,preds,labels=list(range(len(labels))),target_names=labels,zero_division=0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict':saved,'labels':labels,'mean':mean.reshape(9).tolist(),
                'std':std.reshape(9).tolist(),'window':200,'sensors':SENSORS,
                'model_type':'focal_inspired_baseline_v1'},args.output)
    with open(args.output.with_suffix('.metrics.json'),'w',encoding='utf-8') as f:
        json.dump({'val_accuracy':best,'test_report':report,'test_confusion_matrix':confusion_matrix(truth,preds,labels=list(range(len(labels)))).tolist()},f,indent=2)
    print('Saved:',args.output)


if __name__ == '__main__':
    main()
