import cv2
import numpy as np
import pickle
import os

def build_squares(img):
    w, h = 30, 30  # Bigger box size
    d = 10
    start_x = 320 - ((5 * w + 4 * d) // 2)  # Center horizontally
    start_y = 240 - ((10 * h + 9 * d) // 2)  # Center vertically

    crop = None
    positions = []

    y = start_y
    for i in range(10):  # 10 rows
        imgCrop = None
        x = start_x
        for j in range(5):  # 5 columns
            box = img[y:y+h, x:x+w]
            if imgCrop is None:
                imgCrop = box
            else:
                imgCrop = np.hstack((imgCrop, box))
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            positions.append(((x, y), (x+w, y+h)))
            x += w + d
        if crop is None:
            crop = imgCrop
        else:
            crop = np.vstack((crop, imgCrop))
        y += h + d

    return crop, positions

def draw_boxes_on_image(img, positions):
    for pt1, pt2 in positions:
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
    return img

def get_hand_hist():
    save_path = r"C:\Users\Niveditha\Downloads\project2\zip2\hist"
    cam = cv2.VideoCapture(1)
    if not cam.read()[0]:
        cam = cv2.VideoCapture(0)

    flagPressedC = False
    hist = None
    imgCrop = None
    box_positions = []

    while True:
        ret, img = cam.read()
        if not ret:
            break
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord('c'):
            imgCrop, box_positions = build_squares(img.copy())
            hsvCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsvCrop], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
            flagPressedC = True
            print("✅ Histogram captured from green squares.")
        elif keypress == ord('s'):
            if hist is not None:
                with open(save_path, "wb") as f:
                    pickle.dump(hist, f)
                print(f"💾 Histogram saved to: {save_path}")
            break
        elif keypress == ord('q'):
            print("❌ Quit without saving histogram.")
            break

        if flagPressedC:
            dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            cv2.filter2D(dst, -1, disc, dst)
            blur = cv2.GaussianBlur(dst, (11, 11), 0)
            blur = cv2.medianBlur(blur, 15)
            ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh = cv2.merge((thresh, thresh, thresh))
            thresh = draw_boxes_on_image(thresh, box_positions)
            cv2.imshow("Thresh", thresh)
        else:
            _, box_positions = build_squares(img)

        img_with_boxes = draw_boxes_on_image(img, box_positions)
        cv2.imshow("Set hand histogram", img_with_boxes)

    cam.release()
    cv2.destroyAllWindows()

get_hand_hist()
