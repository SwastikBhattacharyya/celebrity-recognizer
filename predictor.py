import joblib
import json
import numpy as np
import cv2
import pywt

model = None
class_names_to_int = None
int_to_class_names = None


def predict(image):
    images = get_cropped_image_if_2_eyes(image)
    result = []
    for image in images:
        scaled_raw_image = cv2.resize(image, (32, 32))
        img_har = w2d(image, 'db1', 5)
        scaled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scaled_raw_image.reshape(32 * 32 * 3, 1), scaled_img_har.reshape(32 * 32, 1)))
        len_image_array = 32 * 32 * 3 + 32 * 32
        final = combined_img.reshape(1, len_image_array).astype(float)
        result.append(
            {
                'class': int_to_class_names[model.predict(final)[0]],
                'class_probability': np.around(model.predict_proba(final) * 100, 2).tolist()[0],
                'class_dictionary': class_names_to_int,
            }
        )
    return result


def load_model():
    global model
    global class_names_to_int
    global int_to_class_names

    model = joblib.load('model.pkl')
    with open('class_dictionary.json', 'r') as f:
        class_names_to_int = json.load(f)
    int_to_class_names = {v: k for k, v in class_names_to_int.items()}


def get_cropped_image_if_2_eyes(image):
    face_cascade = cv2.CascadeClassifier('opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('opencv/haarcascades/haarcascade_eye.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)

    return cropped_faces


def w2d(img, mode='haar', level=1):
    im_array = img
    im_array = cv2.cvtColor(im_array, cv2.COLOR_RGB2GRAY)
    im_array = np.float32(im_array)
    im_array /= 255
    coefficients = pywt.wavedec2(im_array, mode, level=level)

    coefficients_h = list(coefficients)
    coefficients_h[0] *= 0

    im_array_h = pywt.waverec2(coefficients_h, mode)
    im_array_h *= 255
    im_array_h = np.uint8(im_array_h)

    return im_array_h
