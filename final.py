import cv2
import pickle
import numpy as np
import tensorflow as tf
import os
import sqlite3
import pyttsx3
from keras.models import load_model
from threading import Thread

# Suppress OpenCV error/warning messages
try:
	cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except AttributeError:
	os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load model and text-to-speech engine
model = load_model('cnn_model_keras2.h5')
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Globals
x, y, w, h = 300, 100, 300, 300
is_voice_on = True

def get_hand_hist():
	with open("hist", "rb") as f:
		return pickle.load(f)

def get_image_size():
	for folder in os.listdir('gestures'):
		folder_path = os.path.join('gestures', folder)
		if os.path.isdir(folder_path):
			for file in os.listdir(folder_path):
				if file.endswith('.jpg'):
					img = cv2.imread(os.path.join(folder_path, file), 0)
					if img is not None:
						return img.shape
	raise FileNotFoundError("No valid image found in gestures folder")

image_x, image_y = get_image_size()

def keras_process_image(img):
	img = cv2.resize(img, (image_x, image_y))
	img = np.array(img, dtype=np.float32)
	img = np.reshape(img, (1, image_x, image_y, 1))
	return img

def keras_predict(model, image):
	processed = keras_process_image(image)
	pred_probab = model.predict(processed, verbose=0)[0]
	pred_class = np.argmax(pred_probab)
	return max(pred_probab), pred_class

def get_pred_text_from_db(pred_class):
	conn = sqlite3.connect("gesture_db.db")
	cmd = f"SELECT g_name FROM gesture WHERE g_id={pred_class}"
	cursor = conn.execute(cmd)
	text = next(cursor, [None])[0]
	conn.close()
	return text

def get_pred_from_contour(contour, thresh):
	x1, y1, w1, h1 = cv2.boundingRect(contour)
	save_img = thresh[y1:y1+h1, x1:x1+w1]
	if w1 > h1:
		save_img = cv2.copyMakeBorder(save_img, (w1-h1)//2, (w1-h1)//2, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
	else:
		save_img = cv2.copyMakeBorder(save_img, 0, 0, (h1-w1)//2, (h1-w1)//2, cv2.BORDER_CONSTANT, (0, 0, 0))
	pred_probab, pred_class = keras_predict(model, save_img)
	if pred_probab * 100 > 70:
		return get_pred_text_from_db(pred_class) or ""
	return ""

def get_operator(pred_text):
	try:
		pred_text = int(pred_text)
	except:
		return ""
	return {
		1: "+", 2: "-", 3: "*", 4: "/", 5: "%",
		6: "**", 7: ">>", 8: "<<", 9: "&", 0: "|"
	}.get(pred_text, "")

def get_img_contour_thresh(img):
	img = cv2.flip(img, 1)
	imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	dst = cv2.calcBackProject([imgHSV], [0, 1], get_hand_hist(), [0, 180, 0, 256], 1)
	disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
	cv2.filter2D(dst, -1, disc, dst)
	blur = cv2.GaussianBlur(dst, (11, 11), 0)
	blur = cv2.medianBlur(blur, 15)
	thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
	thresh = cv2.merge((thresh, thresh, thresh))
	thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
	thresh = thresh[y:y+h, x:x+w]
	contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
	return img, contours, thresh

def say_text(text):
	if is_voice_on:
		while engine._inLoop:
			pass
		engine.say(text)
		engine.runAndWait()

def calculator_mode(cam):
	global is_voice_on
	flags = {"first": False, "operator": False, "second": False, "clear": False}
	first = operator = second = pred_text = calc_text = ""
	info = "Enter first number"
	Thread(target=say_text, args=(info,)).start()
	count_same_frames = count_clear_frames = 0

	while True:
		ret, img = cam.read()
		if not ret: break
		img = cv2.resize(img, (640, 480))
		img, contours, thresh = get_img_contour_thresh(img)
		old_pred = pred_text
		if contours:
			contour = max(contours, key=cv2.contourArea)
			if cv2.contourArea(contour) > 10000:
				pred_text = get_pred_from_contour(contour, thresh)
				count_same_frames = count_same_frames + 1 if old_pred == pred_text else 0

				if pred_text == "C" and count_same_frames > 5:
					first = second = operator = pred_text = calc_text = ""
					for k in flags: flags[k] = False
					info = "Enter first number"
					Thread(target=say_text, args=(info,)).start()
					count_same_frames = 0

				elif pred_text == "Best of Luck " and count_same_frames > 15:
					if flags["clear"]:
						first = second = operator = pred_text = calc_text = ""
						for k in flags: flags[k] = False
						info = "Enter first number"
					elif second:
						flags["second"] = True
						flags["clear"] = True
						try:
							calc_text += f"= {eval(calc_text)}"
						except:
							calc_text = "Invalid operation"
						Thread(target=say_text, args=(calc_text.replace('**', ' raised to the power ').replace('*', ' multiplied by '),)).start()
					elif first:
						flags["first"] = True
						info = "Enter operator"
						Thread(target=say_text, args=(info,)).start()
					count_same_frames = 0

				elif pred_text.isnumeric():
					if not flags["first"] and count_same_frames > 15:
						first += pred_text
						calc_text += pred_text
						Thread(target=say_text, args=(pred_text,)).start()
						count_same_frames = 0
					elif not flags["operator"]:
						operator = get_operator(pred_text)
						if count_same_frames > 15 and operator:
							flags["operator"] = True
							calc_text += operator
							info = "Enter second number"
							Thread(target=say_text, args=(info,)).start()
							count_same_frames = 0
					elif not flags["second"] and count_same_frames > 15:
						second += pred_text
						calc_text += pred_text
						Thread(target=say_text, args=(pred_text,)).start()
						count_same_frames = 0

		if count_clear_frames == 30:
			first = second = operator = pred_text = calc_text = ""
			for k in flags: flags[k] = False
			info = "Enter first number"
			Thread(target=say_text, args=(info,)).start()
			count_clear_frames = 0

		blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
		cv2.putText(blackboard, "Calculator Mode", (100, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0, 0))
		cv2.putText(blackboard, f"Predicted text- {pred_text}", (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
		cv2.putText(blackboard, f"Operator: {operator}", (30, 140), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 127))
		cv2.putText(blackboard, calc_text, (30, 240), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
		cv2.putText(blackboard, info, (30, 440), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255))
		cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
		cv2.imshow("Recognizing gesture", np.hstack((img, blackboard)))
		cv2.imshow("thresh", thresh)

		key = cv2.waitKey(1)
		if key == ord('q') or key == ord('t'): break
		if key == ord('v'): is_voice_on = not is_voice_on

	return 1 if key == ord('t') else 0

def text_mode(cam):
	global is_voice_on
	text = word = ""
	count_same_frame = 0

	while True:
		ret, img = cam.read()
		if not ret: break
		img = cv2.resize(img, (640, 480))
		img, contours, thresh = get_img_contour_thresh(img)
		old_text = text
		if contours:
			contour = max(contours, key=cv2.contourArea)
			if cv2.contourArea(contour) > 10000:
				text = get_pred_from_contour(contour, thresh)
				count_same_frame = count_same_frame + 1 if old_text == text else 0
				if count_same_frame > 20 and text:
					word += text
					if word.startswith('I/Me '): word = word.replace('I/Me ', 'I ')
					if word.endswith('I/Me '): word = word.replace('I/Me ', 'me ')
					Thread(target=say_text, args=(text,)).start()
					count_same_frame = 0
			elif cv2.contourArea(contour) < 1000 and word:
				Thread(target=say_text, args=(word,)).start()
				text = word = ""
		else:
			if word:
				Thread(target=say_text, args=(word,)).start()
			text = word = ""

		blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
		cv2.putText(blackboard, f"Predicted text- {text}", (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
		cv2.putText(blackboard, word, (30, 240), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
		cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
		cv2.imshow("Recognizing gesture", np.hstack((img, blackboard)))
		cv2.imshow("thresh", thresh)

		key = cv2.waitKey(1)
		if key == ord('q') or key == ord('c'): break
		if key == ord('v'): is_voice_on = not is_voice_on

	return 2 if key == ord('c') else 0

def recognize():
	cam = cv2.VideoCapture(1)
	if not cam.read()[0]:
		cam = cv2.VideoCapture(0)
	keras_predict(model, np.zeros((image_x, image_y), dtype=np.uint8))  # Warmup
	keypress = 1
	while True:
		if keypress == 1:
			keypress = text_mode(cam)
		elif keypress == 2:
			keypress = calculator_mode(cam)
		else:
			break

if __name__ == '__main__':
	recognize()
