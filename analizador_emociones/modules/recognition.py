import cv2
from deepface import DeepFace
import os

class FaceEngine:
    def __init__(self):
        self.db_path = "data"
        if not os.path.exists(self.db_path): os.makedirs(self.db_path)

    def analyze_frame(self, frame):
        try:
            # detector_backend='opencv' es el más rápido para tiempo real
            results = DeepFace.analyze(frame, actions=['emotion'], 
                                    enforce_detection=False, 
                                    detector_backend='opencv', 
                                    silent=True)
            
            name = "No registrado"
            # Solo buscamos si hay fotos registradas
            if len(os.listdir(self.db_path)) > 0:
                dfs = DeepFace.find(frame, db_path=self.db_path, 
                                   model_name="VGG-Face", 
                                   enforce_detection=False, 
                                   detector_backend='opencv',
                                   silent=True)
                if len(dfs) > 0 and not dfs[0].empty:
                    path = dfs[0]['identity'][0]
                    name = os.path.basename(path).split('_')[0]

            return results, name
        except:
            return [], "No registrado"