import cv2, os

def flip_images():
    gest_folder = "gestures"
    for g_id in os.listdir(gest_folder):
        gesture_path = os.path.join(gest_folder, g_id)
        files = os.listdir(gesture_path)
        image_files = [f for f in files if f.endswith(".jpg")]

        for filename in image_files:
            path = os.path.join(gesture_path, filename)
            img = cv2.imread(path, 0)
            if img is None:
                print(f"[ERROR] Couldn't read image: {path}")
                continue

            flipped_img = cv2.flip(img, 1)

            # Save new image with unique name (e.g., "24_flipped.jpg")
            name, ext = os.path.splitext(filename)
            new_filename = f"{name}_flipped{ext}"
            new_path = os.path.join(gesture_path, new_filename)

            cv2.imwrite(new_path, flipped_img)
            print(f"[INFO] Saved flipped image: {new_path}")

flip_images()
