# ANALIZADOR DE EMOCIONES

**Proyecto:** Sistema de reconocimiento y analizador de emociones.

**Autor:** Eldi RAngel

## Descripción

Este proyecto hecho en pythonesta desarollado para que:
- Capture la imagen de la cámara web.
- Registre personas con nombre, apellido y su correo.
- Detecte el rostro y analiza la emoción presente (felicidad, tristeza, enojo, sorpresa, neutral, miedo, disgusto).
- Muestra una interfaz con pestañas de registro, captura y reportes.
- Guarda el historial de detecciones en SQLite.

## Funcionalidades principales

1. **Registro**
   - Capturar y guardar foto de los usuarios.
   - Ingresar nombre/apellido/correo.
   - Control de duplicados por nombre.

2. **funcioes**
   - Iniciar/detener captura desde la cámara.
   - Mostrar recuadros y etiquetas con nombre, emoción y confianza.
   - Guarda cada captura en la base de datos.

3. **Reporte de capturas**
   - Ver historial de detecciones con nombre, emoción, confianza y fecha.
   - Actualizar tabla con los últimos resultados.

## Requisitos

- Python 3.11+
- Biblioteca `opencv-python`
- Biblioteca `deepface`
- Biblioteca `Pillow`
- Biblioteca `tf-keras`
- Biblioteca `pandas`
- Biblioteca `numpy`

## Uso

1. Clona o descarga el proyecto.
2. Instala dependencias:
   
   -pip install -r requirements.txt
   
3. Ejecuta:
   
   -python main.py
   
4. Sigue estos pasos en la app:
   - Registra una persona en la pestaña REGISTRO.
   - Ve a DETECCIÓN y presiona INICIAR DETECCIÓN.
   - Revisa REPORTES y presiona ACTUALIZAR TABLA.

## Estructura de archivos

- `main.py`: Lógica de interfaz Tkinter y flujo principal.
- `modules/recognition.py`: Motor de análisis facial con DeepFace.
- `modules/database_mgr.py`: Manejo de SQLite para historial.
- `requirements.txt`: Dependencias.
