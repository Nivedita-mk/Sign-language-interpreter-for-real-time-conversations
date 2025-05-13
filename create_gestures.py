import cv2
import numpy as np
import pickle, os, sqlite3, random

image_x, image_y = 50, 50

def get_hand_hist():
    with open("hist", "rb") as f:
        hist = pickle.load(f)
    return hist

def init_create_folder_database():
    if not os.path.exists("gestures"):
        os.mkdir("gestures")
    if not os.path.exists("gesture_db.db"):
        conn = sqlite3.connect("gesture_db.db")
        create_table_cmd = "CREATE TABLE gesture ( g_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, g_name TEXT NOT NULL )"
        conn.execute(create_table_cmd)
        conn.commit()
        conn.close()

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

def store_in_db(g_id, g_name):
    conn = sqlite3.connect("gesture_db.db")
    cmd = "INSERT INTO gesture (g_id, g_name) VALUES (?, ?)"
    try:
        conn.execute(cmd, (g_id, g_name))
    except sqlite3.IntegrityError:
        cmd = "UPDATE gesture SET g_name = ? WHERE g_id = ?"
        conn.execute(cmd, (g_name, g_id))
    conn.commit()
    conn.close()

def create_db_from_existing_dataset():
    gesture_path = "gestures"
    subfolders = sorted(os.listdir(gesture_path))
    for g_id, folder_name in enumerate(subfolders):
        folder_path = os.path.join(gesture_path, folder_name)
        if os.path.isdir(folder_path):
            store_in_db(g_id, folder_name)
    print("Database entries added automatically from gestures folder.")

def main():
    init_create_folder_database()
    choice = input("Do you want to add gestures manually to database? (yes/no): ").strip().lower()
    if choice == 'yes':
        g_id = int(input("Enter gesture ID (numeric): "))
        g_name = input("Enter gesture name/text: ")
        store_in_db(g_id, g_name)
        print(f"Entry added manually: ID={g_id}, Name={g_name}")
    else:
        create_db_from_existing_dataset()

if __name__ == '__main__':
    main()
