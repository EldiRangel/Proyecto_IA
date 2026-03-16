import cv2
import os

def capture_face(user_id, frame):
    """Guarda la imagen capturada en la carpeta data/"""
    os.makedirs("data", exist_ok=True)
    file_path = f"data/user_{user_id}.jpg"
    cv2.imwrite(file_path, frame)
    return file_path