from tkinter import *
import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
import numpy as np
import cv2
from tensorflow.lite.python.interpreter import Interpreter
from tkinter.filedialog import askopenfilename


# Lớp chính chứa tất cả các phương thức
class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # Kích hoạt webcam
        self.vid = MyVideoCapture(self.video_source)

        # Lấy các đặc điểm của frame để xây dựng panel
        self.canvas = tkinter.Canvas(window, width=self.vid.width, height=self.vid.height)
        self.canvas.pack(side=LEFT)

        self.xroi = int(np.floor((self.vid.width / 2) - 128))
        self.yroi = int(np.floor((self.vid.height / 2) - 128))

        self.canvas2 = tkinter.Canvas(window, width=512, height=512)
        self.canvas2.pack()

        # Nút để chụp ảnh
        icon1 = PIL.ImageTk.PhotoImage(file="snapshot.png")
        self.btn_snapshot = tkinter.Button(window, image=icon1, width=64, command=self.snapshot)
        self.btn_snapshot.pack(side=LEFT)

        icon2 = PIL.ImageTk.PhotoImage(file="prediction.png")
        self.btn_snapshot = tkinter.Button(window, image=icon2, width=64, command=self.prediction)
        self.btn_snapshot.pack(side=LEFT)

        icon3 = PIL.ImageTk.PhotoImage(file="save.png")
        self.btn_snapshot = tkinter.Button(window, image=icon3, width=64, command=self.saving)
        self.btn_snapshot.pack(side=LEFT)

        # Thêm nút tải ảnh
        icon4 = PIL.ImageTk.PhotoImage(file="upload.png")
        self.btn_upload = tkinter.Button(window, image=icon4, width=64, height=64, command=self.upload_image)
        self.btn_upload.pack(side=LEFT)

        self.interpreter = Interpreter(model_path="MobileNetV2_PlantVillage_Tomato.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']
        self.objects = (
        'Bacterial_Spot',
        'Early_Blight',
        'Late_Blight',
        'Leaf_Mold',
        'Septoria_Leaf_Spot',
        'Spider_Mites',
        'Target_Spot',
        'Yellow_Leaf_Curl_Virus',
        'Mosaic_Virus',
        'Healthy'

        )

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.bottomLeftCornerOfText = (10, 30)
        self.fontScale = 0.6
        self.fontColor = (255,0,0)
        self.lineType = 2

        self.delay = 1
        self.proc = 0
        self.update()

        self.window.mainloop()

    def snapshot(self):
        self.proc = 1

    def prediction(self):
        self.proc = 2

    def saving(self):
        self.proc = 3

    def upload_image(self):
        file_path = askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

        if file_path:
            img = PIL.Image.open(file_path)
            if img.mode == 'RGBA':
                img = img.convert('RGB')

            img_resized = img.resize((224, 224))

            img_array = np.array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = np.float32(img_array) / 255.0

            self.interpreter.set_tensor(self.input_details[0]['index'], img_array)
            self.interpreter.invoke()

            prediction_results = self.interpreter.get_tensor(self.output_details[0]['index'])
            prediction_results = np.array(prediction_results).ravel()

            predicted_class_idx = np.argmax(prediction_results)
            self.pred = self.objects[predicted_class_idx]

            confidence = np.max(prediction_results)
            prediction_text = f"{self.pred} ({confidence:.2f})"

            img_array_resized = np.uint8(img_array[0] * 255)

            cv2.putText(img_array_resized, prediction_text, self.bottomLeftCornerOfText,
                        self.font, self.fontScale, self.fontColor, self.lineType)

            self.photo2 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img_array_resized))
            self.canvas2.create_image(0, 0, image=self.photo2, anchor=tkinter.NW)

            self.ROI = img_array_resized

            self.proc = 0

    def update(self):
        # Lấy frame mới từ webcam
        ret, frame = self.vid.get_frame()
        frame = cv2.rectangle(frame, (self.xroi, self.yroi), (self.xroi + 224, self.yroi + 224), (255, 0, 0), 2)

        if ret:
            if self.proc == 1:  # Trường hợp chụp ảnh
                self.ROI = frame[self.yroi:self.yroi + 224, self.xroi:self.xroi + 224, :]
                self.photo2 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.ROI))
                self.canvas2.create_image(0, 0, image=self.photo2, anchor=tkinter.NW)
                self.proc = 0
            elif self.proc == 2:  # Trường hợp dự đoán
                self.IMG = np.expand_dims(self.ROI, axis=0)
                self.IMG = np.float32(self.IMG) / 255
                self.interpreter.set_tensor(self.input_details[0]['index'], self.IMG)
                self.interpreter.invoke()
                answer = self.interpreter.get_tensor(self.output_details[0]['index'])
                answer = np.array(answer).ravel()
                x = np.argmax(answer)
                self.pred = f"{self.objects[x]} {np.max(answer)}"
                if np.max(answer) > 0.9:  # Mức độ tin cậy cho việc phát hiện [từ -1 đến 1]
                    cv2.putText(self.ROI, self.pred,
                                self.bottomLeftCornerOfText,
                                self.font,
                                self.fontScale,
                                self.fontColor,
                                self.lineType)
                else:
                    self.pred = 'None'
                    cv2.putText(self.ROI, 'None',
                                self.bottomLeftCornerOfText,
                                self.font,
                                self.fontScale,
                                self.fontColor,
                                self.lineType)
                self.photo2 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.ROI))
                self.canvas2.create_image(0, 0, image=self.photo2, anchor=tkinter.NW)
                self.proc = 0
            elif self.proc == 3:
                cv2.imwrite("frame-" + self.pred + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg",
                            cv2.cvtColor(self.ROI, cv2.COLOR_RGB2BGR))
                self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
                self.proc = 0

            else:
                self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))

        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        self.window.after(self.delay, self.update)


class MyVideoCapture:
    def __init__(self, video_source=1):
        # Kiểm tra nguồn video
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Can't find object in video!", video_source)

        # Lấy kích thước video
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Nếu có thể chụp từ webcam, chuyển đổi từ BGR sang RGB
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    # Khi đóng cửa sổ, phải tắt webcam
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# Tạo cửa sổ App
App(tkinter.Tk(), "PlantVillage Tomato disease classifier")