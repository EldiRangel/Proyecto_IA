import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from modules.database_mgr import DatabaseManager
from modules.recognition import FaceEngine
import threading
import os

class FacialApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ANALIZADOR_DE_EMOCIONES")
        self.db = DatabaseManager()
        self.engine = FaceEngine()
        self.cap = cv2.VideoCapture(0)
        
        # Pestañas
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)

        self.tab_reg = ttk.Frame(self.notebook); self.notebook.add(self.tab_reg, text="REGISTRO")
        self.tab_det = ttk.Frame(self.notebook); self.notebook.add(self.tab_det, text="DETECCIÓN")
        self.tab_rep = ttk.Frame(self.notebook); self.notebook.add(self.tab_rep, text="REPORTES")

        self.setup_ui()
        self.is_detecting = False
        self.current_analysis = ([], "sin registrado")
        self.loop()

    def setup_ui(self):
        # UI REGISTRO - Campos Restaurados
        reg_f = ttk.LabelFrame(self.tab_reg, text=" Datos de Registro ")
        reg_f.pack(pady=10, padx=10, fill="x")
        
        ttk.Label(reg_f, text="Nombre:").pack()
        self.ent_nom = ttk.Entry(reg_f); self.ent_nom.pack(pady=2)
        ttk.Label(reg_f, text="Apellido:").pack()
        self.ent_ape = ttk.Entry(reg_f); self.ent_ape.pack(pady=2)
        ttk.Label(reg_f, text="Correo:").pack()
        self.ent_ema = ttk.Entry(reg_f); self.ent_ema.pack(pady=2)
        
        ttk.Button(reg_f, text="REGISTRAR Y CAPTURAR", command=self.registrar).pack(pady=10)
        self.lbl_cam_reg = tk.Label(self.tab_reg); self.lbl_cam_reg.pack()

        # UI DETECCIÓN
        self.lbl_cam_det = tk.Label(self.tab_det); self.lbl_cam_det.pack(pady=10)
        self.btn_det = ttk.Button(self.tab_det, text="INICIAR CAPTURA", command=self.toggle_det)
        self.btn_det.pack(pady=10)

        # UI REPORTES
        self.tree = ttk.Treeview(self.tab_rep, columns=("N","E","C","F"), show="headings")
        self.tree.heading("N", text="Nombre"); self.tree.heading("E", text="Emoción")
        self.tree.heading("C", text="Confianza"); self.tree.heading("F", text="Fecha")
        self.tree.pack(fill="both", expand=True, padx=10, pady=10)
        ttk.Button(self.tab_rep, text="ACTUALIZAR TABLA", command=self.refresh_rep).pack(pady=5)

    def registrar(self):
        nom = self.ent_nom.get().strip()
        ape = self.ent_ape.get().strip()
        ema = self.ent_ema.get().strip()
        if not nom:
            messagebox.showwarning("Atención", "Escribe al menos tu nombre")
            return

        os.makedirs("data", exist_ok=True)
        file_name = f"{nom.replace(' ', '_')}.jpg"
        foto_path = os.path.join("data", file_name)
        if os.path.exists(foto_path):
            messagebox.showwarning("Atención", f"Ya existe un usuario registrado con nombre {nom}.")
            return

        ret, frame = self.cap.read()
        if ret:
            cv2.imwrite(foto_path, frame)
            messagebox.showinfo("Éxito", f"Usuario {nom} registrado correctamente.")

    def toggle_det(self):
        self.is_detecting = not self.is_detecting
        self.btn_det.config(text="DETENER" if self.is_detecting else "INICIAR")
        if self.is_detecting:
            threading.Thread(target=self.run_ai, daemon=True).start()

    def _normalize_analysis(self, analysis):
        """Return a list of face results with consistent structure."""
        if analysis is None:
            return []
        if isinstance(analysis, dict):
            return [analysis]
        if isinstance(analysis, list):
            return analysis
        return []

    def run_ai(self):
        while self.is_detecting:
            ret, frame = self.cap.read()
            if ret:
                self.current_analysis = self.engine.analyze_frame(frame)
                res, name = self.current_analysis
                face_results = self._normalize_analysis(res)
                if len(face_results) > 0:
                    first = face_results[0]
                    emo = first.get('dominant_emotion', 'Desconocida')
                    conf = 0.0
                    emotions = first.get('emotion', {})
                    if isinstance(emotions, dict):
                        conf = float(emotions.get(emo, 0.0))
                    self.db.save_detection(name, emo, conf)

    def refresh_rep(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        stats = self.db.get_stats()
        for r in stats:
            self.tree.insert("", "end", values=(r[0], r[1], f"{r[2]:.2f}", r[3]))

    def loop(self):
        ret, frame = self.cap.read()
        if ret:
            display_frame = frame.copy()
            if self.is_detecting:
                res, name = self.current_analysis
                face_results = self._normalize_analysis(res)
                for r in face_results:
                    region = r.get('region', {})
                    y = int(region.get('y', 0))
                    x = int(region.get('x', 0))
                    w = int(region.get('w', 0))
                    h = int(region.get('h', 0))
                    emotion = r.get('dominant_emotion', 'Desconocida')
                    conf = 0.0
                    emotions = r.get('emotion', {})
                    if isinstance(emotions, dict):
                        conf = float(emotions.get(emotion, 0.0))
                    if w > 0 and h > 0:
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"{name}: {emotion} ({conf:.0f}%)", (x, max(0, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(img))

            # Actualizar cámara en la pestaña de registro y detección
            self.lbl_cam_reg.config(image=img)
            self.lbl_cam_reg.image = img
            self.lbl_cam_det.config(image=img)
            self.lbl_cam_det.image = img

        self.root.after(15, self.loop)

if __name__ == "__main__":
    root = tk.Tk(); app = FacialApp(root); root.mainloop()