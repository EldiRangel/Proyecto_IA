import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

IMG_SIZE = 224
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def load_data(data_dir):
    images = []
    labels = []
    
    print(f"Cargando imagenes desde: {data_dir}")
    for label, class_name in enumerate(CLASSES):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"ADVERTENCIA: No se encuentra la carpeta {class_dir}")
            continue
            
        count = 0
        for file in os.listdir(class_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    images.append(img)
                    labels.append(label)
                    count += 1
        
        print(f"  {class_name}: {count} imagenes cargadas")
    
    return np.array(images), np.array(labels)

if __name__ == "__main__":
    # Prueba de carga
    X, y = load_data('data/dataset-resized')
    print(f"\nTotal imagenes cargadas: {len(X)}")
    print(f"Dimensiones de X: {X.shape}")
    print(f"Dimensiones de y: {y.shape}")
    
    # Dividir en entrenamiento y validacion
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nEntrenamiento: {len(X_train)} imagenes")
    print(f"Validacion: {len(X_val)} imagenes")
