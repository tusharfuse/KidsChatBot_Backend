import sqlite3
import csv
import os

# ---------- CONFIG ----------
DB_PATH = "kidschatbot.db"  # <-- update if your sqlite file name is different
ANIMALS_CSV = "data-1762339147907.csv"
FUNNY_CSV = "data-1762339583691.csv"

# ---------- CONNECT ----------
if not os.path.exists(DB_PATH):
    raise FileNotFoundError(f"SQLite DB not found: {DB_PATH}")

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

print("\n🔄 Resetting tables…")

# ---------- CLEAR TABLES ----------
cur.execute("DELETE FROM Animals;")
cur.execute("DELETE FROM funny_whoami;")
conn.commit()

print("✔ Tables cleared successfully!")

# ===================================================
# 1️⃣ IMPORT ANIMALS QUESTIONS
# ===================================================
print("\n📥 Importing Animals questions…")

with open(ANIMALS_CSV, encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        cur.execute("""
            INSERT INTO Animals (id, question, option_a, option_b, option_c, option_d, image,correct_option,fact)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            int(row["id"]),
            row["question"],
            row["option_a"],
            row["option_b"],
            row["option_c"],
            row["option_d"],
            row.get("image"),
            row["correct_option"],
            row["fact"]
        ))

conn.commit()
print("✔ Animals imported successfully!")

# ===================================================
# 2️⃣ IMPORT FUNNY WHO-AM-I QUESTIONS
# ===================================================
print("\n📥 Importing FunnyWhoami questions…")

with open(FUNNY_CSV, encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        cur.execute("""
            INSERT INTO funny_whoami (id, question, option_a, option_b, option_c, option_d,correct_option,fun_fact)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            int(row["id"]),
            row["question"],
            row["option_a"],
            row["option_b"],
            row["option_c"],
            row["option_d"],
            row["correct_option"],
            row["fun_fact"]
        ))

conn.commit()
print("✔ FunnyWhoami imported successfully!")

conn.close()

print("\n🎉 DONE! Your DB now contains the fresh question sets.")
