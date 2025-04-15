from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Basit kullanıcı veritabanı (örnek)
USERS = {
    "test": "testsifre"
}

TOKENS = {}  # username -> token

# Statik yollar
IMAGE_FOLDER = 'static'
RESULT_FOLDER = 'results'
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Basit token oluşturucu
def generate_token(username):
    return f"token_{username}_12345"

# AUTH endpoint
@app.route('/auth/', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')

    if username in USERS and USERS[username] == password:
        token = generate_token(username)
        TOKENS[username] = token
        return jsonify({'token': token})
    else:
        return jsonify({'detail': 'Invalid credentials'}), 403


# GET frame (token optional for testing)
@app.route('/frames/', methods=['GET'])
def get_frames():
    auth = request.headers.get('Authorization')
    if not validate_token(auth):
        return jsonify({'detail': 'Unauthorized'}), 401

    # Tek test verisi örneği
    response = [
        {
            "url": "http://localhost:5000/frames/1/",
            "image_url": "http://localhost:5000/static/frame_00001.jpg",
            "video_name": "test_session",
            "session": "http://localhost:5000/session/1/",
            "translation_x": 0.01,
            "translation_y": 0.02,
            "translation_z": 0.03,
            "health_status": 1
        },
        {
            "url": "http://localhost:5000/frames/2/",
            "image_url": "http://localhost:5000/static/frame_00002.jpg",
            "video_name": "test_session",
            "session": "http://localhost:5000/session/1/",
            "translation_x": 0.01,
            "translation_y": 0.02,
            "translation_z": 0.03,
            "health_status": 1
        }
    ]
    return jsonify(response)


# Translations örneği
@app.route('/translation/', methods=['GET'])
def get_translation():
    auth = request.headers.get('Authorization')
    if not validate_token(auth):
        return jsonify({'detail': 'Unauthorized'}), 401

    # Örnek veriler
    response = [
        {
            "translation_x": 0.01,
            "translation_y": 0.02,
            "translation_z": 0.03,
            "health_status": 1
        }
    ]
    return jsonify(response)


# Tahmin sonucu al
@app.route('/prediction/', methods=['POST'])
def receive_prediction():
    auth = request.headers.get('Authorization')
    if not validate_token(auth):
        return jsonify({'detail': 'Unauthorized'}), 401

    data = request.get_json()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = os.path.join(RESULT_FOLDER, f'result_{timestamp}.json')
    with open(result_file, 'w') as f:
        json.dump(data, f, indent=4)

    print("📥 Tahmin kaydedildi:", result_file)
    return jsonify({'status': 'ok'}), 201


# Oturum bilgisi
@app.route('/session/<int:id>/', methods=['GET'])
def session_info(id):
    return jsonify({'session_name': f"test_session_{id}"})


# Basit token doğrulayıcı
def validate_token(auth_header):
    if not auth_header:
        return False
    if not auth_header.startswith("Token "):
        return False
    token = auth_header.split(" ")[1]
    return token in TOKENS.values()


@app.route('/')
def index():
    return "TEKNOFEST Test Sunucusu Gelişmiş Versiyon Çalışıyor!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
