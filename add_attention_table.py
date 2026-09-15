"""
add_attention_table.py
======================
Run this file ONCE to add the attention_records table to the database.

How to run (in VS Code Terminal):
    python add_attention_table.py
"""

import sqlite3, os

DB_PATH = os.path.join("db", "attendance.db")

conn = sqlite3.connect(DB_PATH)
conn.execute("""
    CREATE TABLE IF NOT EXISTS attention_records (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id        INTEGER NOT NULL,
        session_date      TEXT    NOT NULL,
        attention_score   REAL    DEFAULT 0.0,
        status            TEXT    DEFAULT 'Attentive',
        duration_seconds  INTEGER DEFAULT 0,
        timestamp         TEXT    NOT NULL,
        FOREIGN KEY (student_id) REFERENCES students (id) ON DELETE CASCADE
    )
""")
conn.commit()
conn.close()
print("attention_records table created successfully!")