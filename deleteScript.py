import os

# 🔥 SET YOUR DATABASE FILE NAME HERE
DB_FILE = "kidschatbot.db"   # <<< change only if your file name is different

def delete_database():
    print("Checking for database file...")

    if os.path.exists(DB_FILE):
        print(f"Deleting database: {DB_FILE}")
        os.remove(DB_FILE)
        print("✅ Database deleted successfully.")
    else:
        print(f"⚠️ No database found named {DB_FILE}. Nothing to delete.")

if __name__ == "__main__":
    delete_database()
