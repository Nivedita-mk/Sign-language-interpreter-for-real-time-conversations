import cv2, os, random
import numpy as np

def get_image_size():
    for gesture in os.listdir('gestures'):
        folder_path = os.path.join('gestures', gesture)
        if not os.path.isdir(folder_path):
            continue
        for img_file in os.listdir(folder_path):
            if img_file.endswith('.jpg'):
                img_path = os.path.join(folder_path, img_file)
                img = cv2.imread(img_path, 0)
                if img is not None:
                    return img.shape
    raise FileNotFoundError("No valid images found in gestures folder.")

gestures = sorted(os.listdir('gestures/'))  # sort alphabetically
begin_index = 0
end_index = 5
image_y, image_x = get_image_size()

rows = len(gestures) // 5 + (1 if len(gestures) % 5 != 0 else 0)
row_images = []

for i in range(rows):
    col_img = None
    for j in range(begin_index, min(end_index, len(gestures))):
        folder_name = gestures[j]
        folder_path = os.path.join("gestures", folder_name)
        image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
        if not image_files:
            img = np.zeros((image_y, image_x), dtype=np.uint8)
        else:
            img_path = os.path.join(folder_path, random.choice(image_files))
            img = cv2.imread(img_path, 0)
            if img is None:
                img = np.zeros((image_y, image_x), dtype=np.uint8)

        col_img = img if col_img is None else np.hstack((col_img, img))

    row_images.append(col_img)
    begin_index += 5
    end_index += 5

# Pad all rows to the same width
max_width = max(row.shape[1] for row in row_images)
for i in range(len(row_images)):
    h, w = row_images[i].shape
    if w < max_width:
        padding = max_width - w
        row_images[i] = cv2.copyMakeBorder(row_images[i], 0, 0, 0, padding, cv2.BORDER_CONSTANT, value=0)

# Now stack vertically
full_img = row_images[0]
for row in row_images[1:]:
    full_img = np.vstack((full_img, row))

cv2.imshow("gestures", full_img)
cv2.imwrite('full_img.jpg', full_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
