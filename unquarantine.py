import sqlite3
import os

# Database path
DB_PATH = "cloud.db"

def release_all_devices():
    """Clear the quarantined_devices table."""
    if not os.path.exists(DB_PATH):
        print(f"Error: Database file '{DB_PATH}' not found.")
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check current count
        cursor.execute("SELECT count(*) FROM quarantined_devices")
        count = cursor.fetchone()[0]
        
        if count == 0:
            print("The 'Cyber Jail' is already empty. No devices are quarantined.")
        else:
            print(f"Found {count} quarantined devices. Releasing them now...")
            cursor.execute("DELETE FROM quarantined_devices")
            conn.commit()
            print("All devices have been released from quarantine successfully!")
            
        conn.close()
    except Exception as e:
        print(f"Error while accessing database: {e}")

if __name__ == "__main__":
    print("-" * 50)
    print("Retro Cloud Anomaly Detection - Quarantine Release")
    print("-" * 50)
    release_all_devices()
    print("-" * 50)
