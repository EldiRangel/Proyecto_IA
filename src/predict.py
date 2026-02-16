
import tensorflow as tf
import cv2
import numpy as np
import sys
import os

# Definir las clases (mismo orden que en preprocess.py)
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
IMG_SIZE = 224

def load_trained_model(model_path='models/trash_classifier.h5'):
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Modelo cargado desde: {model_path}")
        return model
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        print("Asegurate de haber entrenado el modelo primero con train.py")
        sys.exit(1)

def preprocess_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: No existe {image_path}")
        sys.exit(1)
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: No se pudo leer {image_path}")
        sys.exit(1)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

def predict_image(model, image_path):
    img_batch = preprocess_image(image_path)
    predictions = model.predict(img_batch, verbose=0)
    
    predicted_class_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    predicted_class = CLASSES[predicted_class_index]
    
    return predicted_class, confidence, predictions[0]

def main():
    if len(sys.argv) != 2:
        print("Uso: python predict.py <ruta_de_la_imagen>")
        print("Ejemplo: python predict.py data/dataset-resized/plastic/plastic1.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print("Cargando modelo...")
    model = load_trained_model()
    
    print(f"Procesando: {image_path}")
    predicted_class, confidence, all_probs = predict_image(model, image_path)
    
    print("\n" + "="*40)
    print("RESULTADO")
    print("="*40)
    print(f"Clase: {predicted_class}")
    print(f"Confianza: {confidence:.4f} ({confidence*100:.2f}%)")
    print("-"*40)
    print("Probabilidades:")
    for i, class_name in enumerate(CLASSES):
        print(f"  {class_name:10s}: {all_probs[i]:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()