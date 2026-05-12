import sqlite3
import time

def migrate():
    try:
        conn = sqlite3.connect('cloud.db', timeout=10)
        cursor = conn.cursor()
        # Add column if not exists
        try:
            cursor.execute("ALTER TABLE predictions ADD COLUMN device_id VARCHAR DEFAULT 'Unknown'")
            conn.commit()
            print("Successfully added device_id to cloud.db!")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e).lower():
                print("Column device_id already exists.")
            else:
                print(f"Operational error: {e}")
        finally:
            conn.close()
    except Exception as e:
        print(f"Error migrating script: {e}")

if __name__ == "__main__":
    migrate()
