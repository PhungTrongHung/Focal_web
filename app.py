import io
import os
import secrets
from functools import wraps
from flask import Flask, render_template, request, jsonify, session, send_file
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.exceptions import RequestEntityTooLarge
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from processing import clean_csv
from model_adapter import predict

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', secrets.token_hex(32))
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_UPLOAD_MB', '25')) * 1024 * 1024
DEMO_USER = os.getenv('APP_USERNAME', 'PhungTrongHung')
DEMO_PWD_HASH = generate_password_hash(os.getenv('APP_PASSWORD', 'Hung5705'))


def auth_required(fn):
    @wraps(fn)
    def inner(*a, **kw):
        if not session.get('user'):
            return jsonify(error='Vui lòng đăng nhập trước khi dự đoán.'),401
        return fn(*a, **kw)
    return inner

@app.get('/')
def index():
    return render_template('index.html', username=session.get('user'), max_mb=app.config['MAX_CONTENT_LENGTH']//(1024*1024))

@app.post('/api/login')
def login():
    data=request.get_json(silent=True) or {}
    if secrets.compare_digest(str(data.get('username','')).encode('utf-8'), DEMO_USER.encode('utf-8')) and check_password_hash(DEMO_PWD_HASH,str(data.get('password',''))):
        session.clear();session['user']=DEMO_USER
        return jsonify(ok=True,user=DEMO_USER)
    return jsonify(error='Sai tài khoản hoặc mật khẩu.'),401

@app.post('/api/logout')
def logout():
    session.clear();return jsonify(ok=True)

@app.post('/api/predict')
@auth_required
def infer():
    uploaded=request.files.get('file')
    if not uploaded or not uploaded.filename.lower().endswith('.csv'):
        return jsonify(error='Vui lòng chọn tệp .csv.'),400
    try:
        df=clean_csv(uploaded.stream)
        result=predict(df)
        session['last_prediction']=result
        return jsonify(ok=True,**result)
    except (ValueError,RuntimeError,UnicodeDecodeError) as e:
        return jsonify(error=str(e)),422

@app.get('/api/report')
@auth_required
def report():
    r=session.get('last_prediction')
    if not r:return jsonify(error='Chưa có kết quả dự đoán.'),400
    buff=io.BytesIO();c=canvas.Canvas(buff,pagesize=A4)
    c.setFont('Helvetica-Bold',18);c.drawString(55,790,'FOCAL AI - Prediction Report')
    c.setFont('Helvetica',11)
    lines=[f'Model: {r["model_name"]}',f'Prediction: {r["prediction"]}',f'Confidence: {r["confidence"]}%',f'Samples: {r["samples"]}',f'Windows: {r["windows"]}',f'Time span: {r["start_time"]:.2f}s - {r["end_time"]:.2f}s','Class probabilities:']
    lines += [f'  {p["label"]}: {p["probability"]}%' for p in r['probabilities']]
    for i,line in enumerate(lines):c.drawString(55,750-23*i,line)
    c.setFont('Helvetica-Oblique',9);c.drawString(55,70,'Scores are model probabilities, not measured accuracy.')
    c.save();buff.seek(0)
    return send_file(buff,mimetype='application/pdf',as_attachment=True,download_name='focal_ai_report.pdf')

@app.errorhandler(RequestEntityTooLarge)
def too_big(e):return jsonify(error='Tệp vượt quá giới hạn tải lên.'),413

if __name__=='__main__':
    app.run(debug=os.getenv('FLASK_DEBUG')=='1',port=int(os.getenv('PORT','5000')))
