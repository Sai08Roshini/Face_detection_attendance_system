from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import cv2
import numpy as np
import face_recognition
from models import db, User, Attendance
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# Initialize database
db.init_app(app)

# Route: Register User
@app.route('/register_user', methods=['POST'])
def register_user():
    data = request.get_json()
    name = data['name']
    face_image = np.frombuffer(data['face_image'], np.uint8)
    img = cv2.imdecode(face_image, cv2.IMREAD_COLOR)
    
    face_encodings = face_recognition.face_encodings(img)
    
    if len(face_encodings) == 0:
        return jsonify({"error": "No face detected in the image"}), 400
    
    face_encoding = face_encodings[0]
    
    user = User(name=name, face_encoding=face_encoding)
    db.session.add(user)
    db.session.commit()
    
    return jsonify({"message": "User registered successfully", "user_id": user.id})

# Route: Recognize User
@app.route('/recognize_user', methods=['POST'])
def recognize_user():
    face_image = np.frombuffer(request.get_data(), np.uint8)
    img = cv2.imdecode(face_image, cv2.IMREAD_COLOR)
    
    face_encodings = face_recognition.face_encodings(img)
    
    if len(face_encodings) == 0:
        return jsonify({"error": "No face detected in the image"}), 400
    
    encoding = face_encodings[0]
    
    # Compare face with registered users
    users = User.query.all()
    known_encodings = [user.face_encoding for user in users]
    matches = face_recognition.compare_faces(known_encodings, encoding)
    
    if True in matches:
        matched_user = users[matches.index(True)]
        return jsonify({"message": "User recognized", "user_id": matched_user.id, "name": matched_user.name})
    else:
        return jsonify({"error": "User not recognized"}), 404

# Route: Mark Attendance
@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    data = request.get_json()
    user_id = data['user_id']
    
    user = User.query.get(user_id)
    
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    attendance = Attendance(user_id=user_id, status="Present")
    db.session.add(attendance)
    db.session.commit()
    
    return jsonify({"message": "Attendance marked", "user_id": user_id, "timestamp": attendance.timestamp})

if __name__ == '__main__':
    app.run(debug=True)
