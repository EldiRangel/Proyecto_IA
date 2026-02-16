import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# Configuracion
IMG_SIZE = 224
BATCH_SIZE = 8  # pequeño para no saturar memoria
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
EPOCHS = 15  # Menos epocas para ahorrar tiempo/memoria

# Limitar uso de memoria de TensorFlow
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

print("Configurando generadores de datos...")

# Data augmentation basico 
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,  
    zoom_range=0.1,     
    horizontal_flip=True,
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

print("Cargando datos (por lotes)...")

# Cargar datos con batch 
train_generator = train_datagen.flow_from_directory(
    'data/dataset-resized',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',  
    subset='training',
    shuffle=True,
    classes=CLASSES,
    seed=42
)

validation_generator = val_datagen.flow_from_directory(
    'data/dataset-resized',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    classes=CLASSES,
    seed=42
)

print(f"Entrenamiento: {train_generator.samples} imagenes")
print(f"Validacion: {validation_generator.samples} imagenes")

# Modelo mas pequeño y eficiente
print("Construyendo modelo liviano...")
base_model = MobileNetV2(
    weights='imagenet', 
    include_top=False, 
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    alpha=0.5  # Version mas pequeña de MobileNet 
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(32, activation='relu')(x)  
predictions = Dense(len(CLASSES), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Learning rate 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

# Entrenar
print("Iniciando entrenamiento (esto puede tomar tiempo)...")
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    verbose=1,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

# Guardar
os.makedirs('models', exist_ok=True)
model.save('models/trash_classifier_optimized.h5')
print("Modelo guardado!")

# Grafica simple
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validacion')
plt.title('Precision del modelo')
plt.xlabel('Epoca')
plt.ylabel('Precision')
plt.legend()
plt.savefig('models/accuracy_graph.png')
plt.close()  # Cerrar para liberar memoria

print("¡Proceso completado!")