import sqlite3
import os
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path="database/system_db.sqlite"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.create_tables()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def create_tables(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_name TEXT, emocion TEXT, confianza REAL, fecha DATETIME)''')
            conn.commit()

    def save_detection(self, name, emotion, confidence):
        try:
            with self.get_connection() as conn:
                conn.cursor().execute("INSERT INTO history (user_name, emocion, confianza, fecha) VALUES (?,?,?,?)",
                                     (name, emotion, confidence, datetime.now()))
                conn.commit()
        except Exception as e:
            print(f"Error DB: {e}")

    def get_stats(self):
        try:
            with self.get_connection() as conn:
                return conn.cursor().execute("SELECT user_name, emocion, confianza, fecha FROM history ORDER BY fecha DESC LIMIT 50").fetchall()
        except:
            return []