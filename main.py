import telebot
from bot_logic import gen_pass, gen_emodji, flip_coin  # Импортируем функции из bot_logic
import keras
import os
from PIL import Image, ImageOps
import numpy as np
from io import BytesIO

bot = telebot.TeleBot("TOKEN")

# Параметры для обработки изображения
IMAGE_SIZE = (224, 224) 
CLASSES = ['реальный котик', 'нейронка'] 

try:
    model = keras.models.load_model('keras_model.h5')
    print("Модель успешно загружена.") 
except Exception as e:
    print(f"Ошибка при загрузке модели: {e}")



def preprocess_image(image):
    """Предобработка изображения для подачи в модель."""
    size = IMAGE_SIZE
    image = ImageOps.fit(image, size, Image.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32) # Создаем numpy array нужной формы
    data[0] = normalized_image_array
    return data



def predict_image(image):
    """Выполнение предсказания с использованием модели."""
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Я твой Telegram бот. Напиши команду /hello, /bye, /pass, /emodji, /photo или /coin  ")

@bot.message_handler(commands=['hello'])
def send_hello(message):
    bot.reply_to(message, "Шалом! Кем будешь?")

@bot.message_handler(commands=['bye'])
def send_bye(message):
    bot.reply_to(message, "Прощай")

@bot.message_handler(commands=['pass'])
def send_password(message):
    password = gen_pass(10)  # Устанавливаем длину пароля, например, 10 символов
    bot.reply_to(message, f"Вот твой сгенерированный пароль: {password}")

@bot.message_handler(commands=['emodji'])
def send_emodji(message):
    emodji = gen_emodji()
    bot.reply_to(message, f"Вот эмоджи': {emodji}")

@bot.message_handler(commands=['coin'])
def send_coin(message):
    coin = flip_coin()
    bot.reply_to(message, f"Монетка выпала так: {coin}")

@bot.message_handler(regexp="абаюдна")
def greet(message):

  bot.reply_to(message, "И вам абаюдна")


@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    try:
        # Открываем изображение из байтов, используя BytesIO
        image = Image.open(BytesIO(downloaded_file))

        # Получаем предсказание от модели
        prediction = predict_image(image)
        print(f"Prediction: {prediction}")

        # Определяем, что предсказала модель
        predicted_class = CLASSES[np.argmax(prediction)] 
        confidence = prediction[0][np.argmax(prediction)] # Уверенность в предсказании

        # Формируем ответ
        response_text = f"Я думаю, что на этом изображении {predicted_class} (уверенность: {confidence:.2f})"
        bot.reply_to(message, response_text)

    except Exception as e:
        bot.reply_to(message, f"Ошибка при обработке фото: {e}")



# Запускаем бота
bot.polling()
