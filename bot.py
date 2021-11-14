import telebot
from telebot import types
import cv2
import numpy as np
import os.path
import glob
import shutil
import os


file_path = "img/image.jpeg"
token = '2059633155:AAEFOPWn_wA-TpmxCTcXr-hoYsHCoQhNUX8'
bot = telebot.TeleBot(token)


@bot.message_handler(commands=["start"])
def start(m):
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    keyboard.add(*[types.KeyboardButton(name) for name in
                   ['Сотворить магию🔥', 'Информация о проектеℹ', 'Гении проекта🧠']])
    bot.send_message(m.chat.id, '---🎉Artificial Intelligence Telegram Bot🎉---\nОбработка и распознавание объектов на фото и видео. \nВыберите действие:', reply_markup=keyboard)


@bot.message_handler(content_types=['text'])
def message(message):
    if message.text == 'Сотворить магию🔥':
        keyboardgostart = types.ReplyKeyboardMarkup(resize_keyboard=True)
        keyboardgostart.add(*[types.KeyboardButton(name) for name in ['Видео📼', 'Фото🖼']])
        bot.send_message(message.chat.id, 'Выберите Видео или Фото', reply_markup=keyboardgostart)
    elif message.text == 'Гении проекта🧠':
        bot.send_message(message.chat.id, 'Гении данного проекта студенты 1 курса колледжа"РКСИ":\nСбоев Артём: \nVk: vk.com/artem1ka  \nTg: @artemka2604 \nНикита Кульпинов: \nVk: vk.com/drugyourgirl \nTg: @ag1ng ')
    elif message.text == 'Информация о проектеℹ':
        bot.send_message(message.chat.id, "Телеграмм бот на Python'e. Содержит в себе библиотеки: \nOpenCV \nPyTelegramBotApi")
    # Код обработки с видео
    elif message.text == 'Видео📼':
        keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
        keyboard.add(*[types.KeyboardButton(name) for name in ['Обработать Видео📼']])
        bot.send_message(message.chat.id, 'Отправьте мне видео и нажмите на кнопку📼', reply_markup=keyboard)
        if os.path.isfile("video/video.mp4"):
            os.remove("video/video.mp4")
            print("success")
        else:
            print("File doesn't exists!")

        @bot.message_handler(content_types=['video'])
        def video(message):

            bot.send_message(message.chat.id, 'Видео сохранено📼')
            file_info = bot.get_file(message.video.file_id)
            downloaded_file = bot.download_file(file_info.file_path)
            src = file_info.file_path
            with open("video/video.mp4", 'wb') as new_file:
                new_file.write(downloaded_file)

    elif message.text == 'Обработать Видео📼':

        net = cv2.dnn.readNet('artificial_intelligence_files/yolov3.weights',
                              'artificial_intelligence_files/yolov3.cfg')
        classes = []
        with open('artificial_intelligence_files/coco.names', 'r') as f:
            classes = f.read().splitlines()
        cap = cv2.VideoCapture('video/video.mp4')

        try:
            file = open('video/video.mp4')
        except IOError as e:
            keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
            bot.send_message(message.chat.id, 'Отправьте мне видео и нажмите на кнопку📼', reply_markup=keyboard)
        else:
            with file:

                a = 1000000

                length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                length_frime_time = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / 20
                print('Среднее время выполнения: ' + str(round(length_frime_time)) + ' минут.')

                bot.send_message(message.chat.id, 'Среднее время выполнения: ' + str(round(length_frime_time)) + ' мин.')

                while True:
                    length -= 1
                    _, img = cap.read()
                    height, width, _ = img.shape

                    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
                    net.setInput(blob)
                    output_layers_names = net.getUnconnectedOutLayersNames()
                    layerOutputs = net.forward(output_layers_names)

                    boxes = []
                    confidences = []
                    class_ids = []

                    for output in layerOutputs:
                        for detection in output:
                            scores = detection[5:]
                            class_id = np.argmax(scores)
                            confidence = scores[class_id]
                            if confidence > 0.5:
                                center_x = int(detection[0] * width)
                                center_y = int(detection[1] * height)
                                w = int(detection[2] * width)
                                h = int(detection[3] * height)

                                x = int(center_x - w / 2)
                                y = int(center_y - h / 2)

                                boxes.append([x, y, w, h])
                                confidences.append((float(confidence)))
                                class_ids.append(class_id)

                    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

                    font = cv2.FONT_HERSHEY_PLAIN
                    colors = np.random.uniform(0, 255, size=(len(boxes), 3))
                    if len(indexes) > 0:
                        for i in indexes.flatten():
                            x, y, w, h = boxes[i]
                            label = str(classes[class_ids[i]])
                            confidence = str(round(confidences[i], 2))
                            color = colors[i]
                            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                            cv2.putText(img, label + " " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)

                    cv2.imwrite('video/video_cadr/' + str(a) + '.jpg', img)
                    a += 1
                    if length < 1:
                        break
                img_array = []
                for filename in glob.glob('video/video_cadr/*.jpg'):
                    img = cv2.imread(filename)
                    height, width, layers = img.shape
                    size = (width, height)
                    img_array.append(img)

                out = cv2.VideoWriter('project.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 27, size)

                for i in range(len(img_array)):
                    out.write(img_array[i])


                shutil.rmtree("video\\video_cadr", ignore_errors=True)
                path = "video\\video_cadr"
                os.mkdir(path)
                keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
                keyboard.add(*[types.KeyboardButton(name) for name in ['Получить Видео']])
                bot.send_message(message.chat.id, 'Видео обработано', reply_markup=keyboard)

    elif message.text == 'Получить Видео':
        bot.send_video(message.chat.id, open("project.mp4", 'rb'))


    # Фото распознавание
    elif message.text == 'Фото🖼':
        keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
        keyboard.add(*[types.KeyboardButton(name) for name in ['Обработать Фото🖼']])
        bot.send_message(message.chat.id, 'Отправьте мне фото и нажмите на кнопку🖼', reply_markup=keyboard)
        if os.path.isfile("img/image.jpeg"):
            os.remove("img/image.jpeg")
            print("success")
        else:
            print("File doesn't exists!")

        @bot.message_handler(content_types=['photo'])
        def handle(message):
            bot.send_message(message.chat.id, 'Фото сохранено🖼')
            fileID = message.photo[-1].file_id
            file = bot.get_file(fileID)
            down_file = bot.download_file(file.file_path)
            with open("img/image.jpeg", "wb") as f:
                f.write(down_file)

    elif message.text == 'Обработать Фото🖼':
            try:
                file = open('img/image.jpeg')
            except IOError as e:
                keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
                bot.send_message(message.chat.id, 'Отправьте мне фото и нажмите на кнопку🖼', reply_markup=keyboard)
            else:
                with file:

                    net = cv2.dnn.readNet('artificial_intelligence_files/yolov3.weights',
                                          'artificial_intelligence_files/yolov3.cfg')
                    classes = []
                    with open('artificial_intelligence_files/coco.names', 'r') as f:
                        classes = f.read().splitlines()

                        img = cv2.imread('img/image.jpeg')
                        height, width, _ = img.shape

                        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
                        net.setInput(blob)
                        output_layers_names = net.getUnconnectedOutLayersNames()
                        layerOutputs = net.forward(output_layers_names)

                        boxes = []
                        confidences = []
                        class_ids = []

                        for output in layerOutputs:
                            for detection in output:
                                scores = detection[5:]
                                class_id = np.argmax(scores)
                                confidence = scores[class_id]
                                if confidence > 0.5:
                                    center_x = int(detection[0] * width)
                                    center_y = int(detection[1] * height)
                                    w = int(detection[2] * width)
                                    h = int(detection[3] * height)

                                    x = int(center_x - w / 2)
                                    y = int(center_y - h / 2)

                                    boxes.append([x, y, w, h])
                                    confidences.append((float(confidence)))
                                    class_ids.append(class_id)

                        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

                        font = cv2.FONT_HERSHEY_PLAIN
                        colors = np.random.uniform(0, 255, size=(len(boxes), 3))

                        for i in indexes.flatten():
                            x, y, w, h = boxes[i]
                            label = str(classes[class_ids[i]])
                            confidence = str(round(confidences[i], 2))
                            color = colors[i]
                            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                            cv2.putText(img, label + " " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)

                        cv2.imwrite("img/img_finished.jpeg", img)
                bot.send_photo(message.chat.id, open("img/img_finished.jpeg", 'rb'))

bot.polling(none_stop=True)
