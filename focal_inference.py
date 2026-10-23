from pathlib import Path
import numpy as np
import torch
from focal_model import FocalModel, SENSORS

CHECKPOINT = Path(__file__).resolve().parent / 'models' / 'focal.pt'

def predict_focal(df, checkpoint=CHECKPOINT):
    if not Path(checkpoint).is_file():
        raise RuntimeError('Chưa có models/focal.pt. Hãy huấn luyện bằng train_focal.py với CSV đầy đủ (không dùng head30).')
    obj = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if obj.get('model_type') != 'focal_inspired_baseline_v1' or obj.get('sensors') != SENSORS:
        raise ValueError('Checkpoint không tương thích với mô hình FOCAL baseline này.')
    labels = obj['labels']
    model = FocalModel(len(labels))
    model.load_state_dict(obj['state_dict'])
    model.eval()
    window = int(obj['window'])
    vals = df[SENSORS].to_numpy(dtype=np.float32)
    if len(vals) < window:
        raise ValueError(f'Cần ít nhất {window} dòng dữ liệu hợp lệ.')
    mean=np.asarray(obj['mean'], dtype=np.float32)
    std=np.asarray(obj['std'], dtype=np.float32)
    if np.any(std <= 0):
        raise ValueError('Tham số chuẩn hóa không hợp lệ.')
    # Non-overlap and batched inference to limit memory usage.
    usable=(len(vals)//window)*window
    view=vals[:usable].reshape(-1,window,len(SENSORS))
    running=np.zeros(len(labels),dtype=np.float64)
    per_label={k:0 for k in labels}
    with torch.inference_mode():
        for start in range(0,len(view),128):
            xs=((view[start:start+128]-mean)/std).astype(np.float32)
            prob=torch.softmax(model(torch.from_numpy(xs)),dim=-1).cpu().numpy()
            running+=prob.sum(axis=0)
            for class_index in prob.argmax(axis=1):
                per_label[labels[int(class_index)]]+=1
    avg=running/len(view)
    results=sorted(({'label': label,'probability':round(float(prob)*100,2)} for label,prob in zip(labels,avg)),key=lambda x:x['probability'],reverse=True)
    return {
        'prediction':results[0]['label'],
        'confidence':results[0]['probability'],
        'probabilities':results,
        'windows':len(view), 'samples':len(df),
        'model_name':'FOCAL-inspired PyTorch baseline (không phải bản FOCAL gốc)',
        'start_time':float(df['time_s'].iloc[0]),
        'end_time':float(df['time_s'].iloc[-1]),
        'window_predictions':per_label,
    }
