import cv2
import numpy as np
import tensorflow as tf

# cargar el modelo entrenado
print("cargando modelo...")
modelo = tf.keras.models.load_model("models/trash_classifier_optimized.h5")
print("modelo cargado correctamente")

# clases del dataset
clases = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# abrir la camara
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("ERROR: No se pudo abrir la camara")
    exit()

print("\n=== CLASIFICADOR DE RESIDUOS===")
print("La camara se abrira en una ventana")
print("Presiona 'q' para salir")
print("Presiona 's' para guardar captura")
print("===========================================\n")

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: No se pudo leer el frame")
        break

    # preprocesar imagen
    img = cv2.resize(frame, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # prediccion
    pred = modelo.predict(img, verbose=0)
    clase_idx = np.argmax(pred)
    prob = np.max(pred)

    # confianza
    if prob > 0.8:
        color = (0, 255, 0)  # verde (alta confianza)
    elif prob > 0.6:
        color = (0, 255, 255)  # amarillo (media confianza)
    else:
        color = (0, 0, 255)  # rojo (baja confianza)

    texto = f"{clases[clase_idx]}: {prob:.2%}"
    
    # mostrar texto en pantalla
    cv2.putText(frame, texto, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # mostrar todas las clases 
    y_pos = 60
    for i, clase in enumerate(clases):
        prob_text = f"{clase}: {pred[0][i]:.2%}"
        cv2.putText(frame, prob_text, (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 20

    cv2.imshow("Clasificador de Residuos - IA", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        # guardar captura
        filename = f"captura_{clases[clase_idx]}_{int(prob*100)}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Captura guardada: {filename}")

cam.release()
cv2.destroyAllWindows()
print("Programa terminado")