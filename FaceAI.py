import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from fer import FER
import pytesseract
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import dlib
import matplotlib.pyplot as plt
import datetime

#بارگذاری مدل InceptionV3 برای شناسایی ویژگی‌ها
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  #طبقه‌بندی دوگانه
])
# بارگذاری مدل شناسایی احساسات از FER
emotion_detector = FER()
# بارگذاری مدل‌های شناسایی چهره dlib
face_detector = dlib.get_frontal_face_detector()

# تشخیص و آنالیز چهره‌ها
def detect_faces(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # شناسایی چهره‌ها با استفاده از dlib
    faces = face_detector(gray)
    if len(faces) == 0:
        print("چهره‌ای شناسایی نشد.")
        return img, []
    
    return img, faces

# شناسایی احساسات چهره
def recognize_face_and_emotions(image_path):
    img, faces = detect_faces(image_path)
    if len(faces) == 0:
        print("چهره‌ای برای شناسایی ویژگی‌ها یافت نشد.")
        return
    
    for face in faces:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        #برش چهره از تصویر
        face_img = img[y:y+h, x:x+w]
        # تغییر اندازه چهره به ابعاد مورد نیاز مدل
        face_img_resized = cv2.resize(face_img, (299, 299))
        face_img_resized = np.expand_dims(face_img_resized, axis=0)
        face_img_resized = preprocess_input(face_img_resized)
        #پیش‌بینی ویژگی‌ها با مدل InceptionV3
        features = model.predict(face_img_resized)
        print(f"ویژگی‌های شناسایی‌شده برای چهره: {features}")
        #احساسات چهره با FER
        emotion, score = emotion_detector.top_emotion(face_img_resized)
        print(f"احساسات شناسایی‌شده: {emotion} با امتیاز: {score}")
        #تحلیل ویژگی‌های صورت
        eye_aspect_ratio = calculate_eye_aspect_ratio(face_img)
        print(f"نسبت ابعاد چشم: {eye_aspect_ratio}")
        # نمایش چهره شناسایی‌شده
        plt.imshow(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        plt.show()

    #نمایش تصویر نهایی
    cv2.imshow('Detected Faces', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# تشخیص و آنالیز دست‌خط‌ها با استفاده از OCR
def recognize_handwriting(image_path):
    img = cv2.imread(image_path)
    # پردازش OCR برای تشخیص متن
    text = pytesseract.image_to_string(img)
    if text:
        print("متن شناسایی‌شده از دست‌خط: ")
        print(text)
    else:
        print("هیچ متنی شناسایی نشد.")

#پردازش و آنالیز ویدیو
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        #شناسایی چهره‌ها
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)
        
        for face in faces:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #نمایش ویدیو در حین پردازش
        cv2.imshow('Video Face Detection', frame)
        #خروج از ویدیو با کلید ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

#محاسبه نسبت ابعاد چشم (Eye Aspect Ratio) برای تحلیل ویژگی‌های صورت
def calculate_eye_aspect_ratio(face_img):
    #پردازش برای شناسایی چشمان
    #استفاده از مدل‌هایی مانند Dlib یا OpenCV برای شناسایی چشمان
    #در اینجا صرفاً یک مقدار فرضی برای توضیح آورده شده است
    return 0.25

#  ثبت سوابق برای هر تصویر
def log_analysis_results(image_path, emotion, score):
    now = datetime.datetime.now()
    log_message = f"تحلیل تصویر {image_path} در تاریخ {now}: احساسات {emotion} با امتیاز {score}\n"
    with open("analysis_log.txt", "a") as log_file:
        log_file.write(log_message)

#ربات تلگرامی برای دریافت تصویر و ارسال تحلیل‌ها
def start(update: Update, context: CallbackContext):
    update.message.reply_text('سلام! لطفاً تصویری ارسال کنید تا آنالیز شود.')

def handle_image(update: Update, context: CallbackContext):
    # دریافت تصویر از کاربر
    photo = update.message.photo[-1]
    photo_file = photo.get_file()
    image_path = 'user_image.jpg'
    photo_file.download(image_path)
    
    update.message.reply_text('تصویر دریافت شد، در حال آنالیز...')
    #آنالیز چهره‌ها و احساسات
    recognize_face_and_emotions(image_path)
    #آنالیز دست‌خط
    recognize_handwriting(image_path)
    #ثبت نتایج
    emotion, score = emotion_detector.top_emotion(image_path)
    log_analysis_results(image_path, emotion, score)
    update.message.reply_text('آنالیز تصویر به پایان رسید.')

def handle_video(update: Update, context: CallbackContext):
    #دریافت ویدیو از کاربر
    video = update.message.video
    video_file = video.get_file()
    video_path = 'user_video.mp4'
    video_file.download(video_path)
    update.message.reply_text('ویدیو دریافت شد، در حال آنالیز...')
    #آنالیز ویدیو
    process_video(video_path)
    update.message.reply_text('آنالیز ویدیو به پایان رسید.')

#راه‌اندازی ربات تلگرام
def main():
    updater = Updater('YOUR_BOT_API_KEY', use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.photo, handle_image))
    dp.add_handler(MessageHandler(Filters.video, handle_video))
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
