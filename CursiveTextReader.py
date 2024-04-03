import tensorflow as tf
from tensorflow.keras import models
import cv2
import numpy as np

def decode_prediction(prediction):
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    predicted_text = ""
    for probs in prediction:
        char_index = np.argmax(probs)
        predicted_text += alphabet[char_index]
    return predicted_text

# Загрузка обученной модели распознавания текста из TensorFlow
model = models.load_model('cursive_text_reader_model.h5')

# Загрузка изображения с текстом
image_path = r"C:\Users\kiris\Downloads\2024-03-27_14-52-01.png"
print("Image path:", image_path)

image = cv2.imread(image_path)
if image is None:
    print("Error: Image not found!")
else:
    print("Image loaded successfully!")

# Преобразование изображения в оттенки серого и нормализация значений пикселей
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
normalized = gray / 255.0

# Изменение размера изображения до размера, ожидаемого моделью
input_size = (640, 360)  
resized_image = cv2.resize(normalized, input_size)

# Расширение размерности изображения для подачи в модель
input_data = np.expand_dims(np.expand_dims(resized_image, axis=0), axis=-1)

# Предсказание текста на изображении с помощью модели
prediction = model.predict(input_data)
predicted_text = decode_prediction(prediction)

# Вывод распознанного текста на терминале
print("Predicted text:", predicted_text)
