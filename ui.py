import detect
import sys
import os
import cv2
import time
import numpy as np
import torch
from threading import Thread
from models.common import DetectMultiBackend
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QLabel, QPushButton, \
    QFileDialog, QTextEdit, QComboBox, QRadioButton, QCheckBox, QLineEdit, QMessageBox
from PyQt5.QtGui import QDoubleValidator, QPainter, QPen, QIcon, QIntValidator, QPixmap, QPalette, QBrush
from PyQt5.QtCore import Qt, QUrl, QTimer


class Parameters:
    def __init__(self):
        self.model = ''
        self.data_path = ''
        self.save_path = ''
        self.NMS_threshold = 0.5
        self.confidence_threshold = 0.5
        self.max = 10
        self.hide_confidence = False
        self.hide_label = False


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        # 菜单栏设置
        self.menuBar = self.menuBar()
        self.menuBar.addAction("File")
        self.menuBar.addAction("View")

        # 中央组件设置
        self.window = MainWidget(self)
        self.setCentralWidget(self.window)

        # 状态栏设置
        self.statusBar = self.statusBar()
        self.statusBar.showMessage("", 5000)
        label = QLabel("permanent status")
        self.statusBar.addPermanentWidget(label)

        self.resize(1000, 800)
        self.setWindowTitle("Black smoke vehicle detector")
        self.setWindowIcon(QIcon('./ico.ico'))

        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(QPixmap("./test2.jpg").scaled(self.width(), self.height())))
        self.setPalette(palette)


class MainWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.par = Parameters()
        self.i = 0
        self.j = 0
        self.file_list = []
        self.save_dir = ''
        self.open_file = False
        self.is_realtime = False
        self.result = False
        self.model = ''
        self.time = QTimer()
        self.time_video = QTimer()
        self.time.timeout.connect(self.show_realtime_img)
        self.time_video.timeout.connect(self.confidence_of_video)
        self.realtime_img = ''
        self.cap = cv2.VideoCapture()
        self.fps = 25
        self.label_data = QLabel('文件路径', self)
        self.label_data.move(20, 605)
        self.label_save = QLabel('保存路径', self)
        self.label_save.move(20, 645)
        self.label_model = QLabel('模型路径', self)
        self.label_model.move(20, 565)
        self.label_pic_or_mov = QLabel('数据形式', self)
        self.label_pic_or_mov.move(10, 22)
        self.label_mode = QLabel('检测模式', self)
        self.label_mode.move(10, 52)
        self.label_realtime = QLabel('实时检测', self)
        self.label_realtime.move(10, 82)
        self.label_hide_confidence = QLabel('隐藏置信度', self)
        self.label_hide_confidence.move(10, 112)
        self.label_hide_confidence = QLabel('隐藏标签', self)
        self.label_hide_confidence.move(10, 142)
        self.label_NMS = QLabel('NMS阈值', self)
        self.label_NMS.move(10, 172)
        self.label_confidence = QLabel('置信阈值', self)
        self.label_confidence.move(10, 202)
        self.label_max_target = QLabel('最大数目', self)
        self.label_max_target.move(10, 232)
        self.label_is_smoke = QLabel('黑烟', self)
        self.label_is_smoke.move(10, 300)
        self.label_is_smoke_ = QLabel(self)
        self.label_is_smoke_.move(90, 300)
        self.label_is_smoke_.resize(100, 15)
        self.label_show_confidence = QLabel('置信度', self)
        self.label_show_confidence.move(10, 330)
        self.label_show_confidence_ = QLabel(self)
        self.label_show_confidence_.move(90, 330)
        self.label_show_confidence_.resize(100, 15)
        self.label_start_time = QLabel('起始时间', self)
        self.label_start_time.move(10, 360)
        self.label_start_time_ = QLabel(self)
        self.label_start_time_.move(90, 360)
        self.label_start_time_.resize(100, 15)
        self.label_end_time = QLabel('结束时间', self)
        self.label_end_time.move(10, 390)
        self.label_end_time_ = QLabel(self)
        self.label_end_time_.move(90, 390)
        self.label_end_time_.resize(100, 15)
        self.label_detect_time = QLabel('检测时间', self)
        self.label_detect_time.move(10, 420)
        self.label_detect_time_ = QLabel(self)
        self.label_detect_time_.move(80, 420)
        self.label_detect_time_.resize(200, 15)
        self.par.model = 'E:/detector/station.pt'
        self.label_realtime_detect = QLabel(self)
        self.label_realtime_detect.resize(800, 550)
        self.label_realtime_detect.move(200, 0)
        self.label_message = QLabel(self)
        self.label_message.move(20, 680)
        self.label_message.resize(100, 30)

        doublevalidator = QDoubleValidator(self)
        doublevalidator.setRange(0, 1)
        doublevalidator.setDecimals(2)
        doublevalidator.setNotation(QDoubleValidator.StandardNotation)

        self.lineEdit_NMS = QLineEdit(self)
        self.lineEdit_NMS.move(80, 170)
        self.lineEdit_NMS.resize(40, 20)
        self.lineEdit_NMS.setText('0.5')
        self.lineEdit_NMS.setValidator(doublevalidator)

        self.lineEdit_confidence = QLineEdit(self)
        self.lineEdit_confidence.move(80, 200)
        self.lineEdit_confidence.resize(40, 20)
        self.lineEdit_confidence.setText('0.5')
        self.lineEdit_confidence.setValidator(doublevalidator)

        self.lineEdit_max_target = QLineEdit(self)
        self.lineEdit_max_target.move(80, 230)
        self.lineEdit_max_target.resize(40, 20)
        self.lineEdit_max_target.setText('10')
        self.lineEdit_max_target.setValidator(QIntValidator(1, 1000))

        self.text_data = QTextEdit(self)
        self.text_data.resize(600, 30)
        self.text_data.move(100, 600)
        self.text_save = QTextEdit(self)
        self.text_save.resize(600, 30)
        self.text_save.move(100, 640)
        self.text_model = QTextEdit(self)
        self.text_model.resize(600, 30)
        self.text_model.move(100, 560)
        self.text_model.setPlainText(self.par.model)

        self.button_get_data_path = QPushButton('open', self)
        self.button_get_data_path.move(750, 600)
        self.button_get_data_path.clicked.connect(self.getdata)

        self.button_get_save_path = QPushButton('save', self)
        self.button_get_save_path.move(750, 640)
        self.button_get_save_path.clicked.connect(self.save)

        self.button_get_model_path = QPushButton('model', self)
        self.button_get_model_path.move(750, 560)
        self.button_get_model_path.clicked.connect(self.getmodel)

        self.button_detect = QPushButton('detect', self)
        self.button_detect.resize(110, 110)
        self.button_detect.move(870, 560)
        self.button_detect.clicked.connect(self.start_detect)
        self.button_detect.setEnabled(False)

        self.button_before = QPushButton('上一张', self)
        self.button_before.resize(60, 30)
        self.button_before.move(10, 450)
        self.button_before.setEnabled(True)
        self.button_before.clicked.connect(self.i_decrease)

        self.button_last = QPushButton('下一张', self)
        self.button_last.resize(60, 30)
        self.button_last.move(10, 500)
        self.button_last.setEnabled(True)
        self.button_last.clicked.connect(self.i_plus)

        self.button_start = QPushButton('开始', self)
        self.button_start.resize(60, 30)
        self.button_start.move(100, 450)
        self.button_start.setEnabled(False)
        self.button_start.clicked.connect(self.play_video)

        self.button_end = QPushButton('暂停', self)
        self.button_end.resize(60, 30)
        self.button_end.move(100, 500)
        self.button_end.setEnabled(False)
        self.button_end.clicked.connect(self.pause_video)

        self.button_realtime = QRadioButton('', self)
        self.button_realtime.move(100, 82)
        self.button_realtime.toggled.connect(self.change_realtime)

        self.button_hide_confidence = QCheckBox('', self)
        self.button_hide_confidence.move(100, 112)
        self.button_hide_confidence.stateChanged.connect(self.hide_confidence)

        self.button_hide_label = QCheckBox('', self)
        self.button_hide_label.move(100, 142)
        self.button_hide_label.stateChanged.connect(self.hide_label)

        self.detect_mode = QComboBox(self)
        self.detect_mode.addItem('station')
        self.detect_mode.addItem('road')
        self.detect_mode.move(80, 50)
        self.detect_mode.currentIndexChanged.connect(self.change_mode)

        self.pic = QComboBox(self)
        self.pic.addItem('图片')
        self.pic.addItem('视频')
        self.pic.move(80, 20)
        self.pic.currentIndexChanged.connect(self.mov)

        self.player = QMediaPlayer()
        self.vw = QVideoWidget(self)
        self.vw.resize(800, 550)
        self.vw.move(200, 0)
        self.vw.show()
        self.player.setVideoOutput(self.vw)
        self.player.setMedia(QMediaContent(QUrl('file:///E:/detector/bg.jpg')))
        self.player.play()

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        pen = QPen(Qt.red, 1, Qt.SolidLine)
        painter.setPen(pen)
        painter.drawLine(0, 550, 1000, 550)
        painter.drawLine(0, 270, 200, 270)
        painter.drawLine(200, 0, 200, 550)

    def getdata(self):
        self.par.data_path = QFileDialog.getExistingDirectory(self, "选取待检测文件", "C:/")
        self.text_data.setPlainText(self.par.data_path)
        if self.text_model.toPlainText() != '' and self.label_message.text() != '检测中' \
                and self.text_data.toPlainText() != '' and self.text_save.toPlainText() != '':
            self.button_detect.setEnabled(True)

    def save(self):
        self.par.save_path = QFileDialog.getExistingDirectory(self, "保存为", "C:/")
        self.text_save.setPlainText(self.par.save_path)
        if self.text_model.toPlainText() != '' and self.label_message.text() != '检测中' \
                and self.text_data.toPlainText() != '' and self.text_save.toPlainText() != '':
            self.button_detect.setEnabled(True)
        if self.is_realtime and self.text_model.toPlainText() != '' and self.text_save.toPlainText() != '':
            self.button_detect.setEnabled(True)

    def getmodel(self):
        self.par.model = QFileDialog.getOpenFileName(self, "选取模型", "C:/", "model (*.pt)")[0]
        self.text_model.setPlainText(self.par.model)
        if self.text_model.toPlainText() != '' and self.label_message.text() != '检测中'\
                and self.text_data.toPlainText() != '' and self.text_save.toPlainText() != '':
            self.button_detect.setEnabled(True)
        if self.is_realtime and self.text_model.toPlainText() != '' and self.text_save.toPlainText() != '':
            self.button_detect.setEnabled(True)

    def change_mode(self):
        if self.text_model.toPlainText() == 'E:/detector/station.pt' or self.text_model.toPlainText() == 'E:/detector/road.pt':
            if self.detect_mode.currentText() == 'station':
                self.text_model.setPlainText('E:/detector/station.pt')
            else:
                self.text_model.setPlainText('E:/detector/road.pt')

    def mov(self):
        self.button_before.setEnabled(False)
        self.button_last.setEnabled(False)
        self.button_start.setEnabled(False)
        self.button_end.setEnabled(False)

    def change_realtime(self):
        self.is_realtime = not self.is_realtime
        if len(self.text_save.toPlainText()) != 0 and self.label_message.text() != '检测中' \
                and len(self.text_model.toPlainText()) != 0 and self.is_realtime:
            self.button_detect.setEnabled(True)
        elif self.text_model.toPlainText() != '' and self.label_message.text() != '检测中' \
                and self.text_save.toPlainText() != '' and self.text_data.toPlainText() != '':
            self.button_detect.setEnabled(True)
        else:
            self.button_detect.setEnabled(False)
        if self.is_realtime:
            self.label_message.setText('摄像头开启')
            self.button_last.setEnabled(False)
            self.button_before.setEnabled(False)
        else:
            self.label_message.setText('摄像头关闭')
            self.button_start.setEnabled(False)
            self.button_end.setEnabled(False)

    def hide_confidence(self):
        self.par.hide_confidence = not self.par.hide_confidence

    def hide_label(self):
        self.par.hide_label = not self.par.hide_label

    def play_video(self):
        self.player.play()
        if not self.is_realtime:
            self.time_video.start(int(1000 / self.fps))

    def pause_video(self):
        self.player.pause()
        self.time_video.stop()
        if self.is_realtime:
            self.button_realtime.setEnabled(True)
            self.button_before.setEnabled(False)
            self.button_last.setEnabled(False)
            self.button_start.setEnabled(True)
            self.button_end.setEnabled(True)
            self.time.stop()
            self.cap.release()
            path = self.par.save_path + '/' + 'detect'
            self.vw.show()
            self.player.setMedia(QMediaContent(QUrl('file:///E:/detector/bg.jpg')))
            self.player.play()
            smoke = 0
            if os.path.exists(path):
                for root, dirs, files in os.walk(self.par.save_path + '/' + 'detect'):
                    ls = []
                    for file in files:
                        splitext = os.path.splitext(file)
                        if splitext[1] == '.jpg':
                            fname = os.path.join(root, file)
                            ls.append(fname)
                    if len(ls):
                        video_path = self.par.save_path + '/' + 'detect' + '/' + "result.mp4"
                        if not os.path.exists(video_path):
                            fps = 12
                            size = cv2.imread(ls[0]).shape[1::-1]
                            videowriter = cv2.VideoWriter(video_path, -1, fps, size)
                            for x in ls:
                                img = cv2.imread(x)
                                videowriter.write(img)
                            videowriter.release()
                            if os.path.exists(self.par.save_path + '/detect/labels'):
                                for root, dirs, files in os.walk(self.par.save_path + '/detect/labels'):
                                    for file in files:
                                        _, splitext = os.path.splitext(file)
                                        if splitext == '.txt':
                                            smoke = 1
                                            break
                            button = QMessageBox.question(self, "Question", "是否打开检测结果",
                                                          QMessageBox.Yes | QMessageBox.No,
                                                          QMessageBox.Yes)
                            self.label_message.setText('检测完毕')
                            if button == QMessageBox.Yes:
                                self.player.setMedia(QMediaContent(QUrl('file:///' + video_path)))
                                self.player.play()
                                self.label_show_confidence_.setText('--')
                                if smoke:
                                    self.label_is_smoke_.setText('Yes')
                                else:
                                    self.label_is_smoke_.setText('No')
                            else:
                                self.player.setMedia(QMediaContent(QUrl('file:///E:/detector/bg.jpg')))
                                self.player.play()
            self.pic.setEnabled(True)
            if self.text_model.toPlainText() != '' and self.label_message.text() != '检测中' \
                    and self.text_data.toPlainText() != '' and self.text_save.toPlainText() != '':
                self.button_detect.setEnabled(True)
            if self.is_realtime and self.text_model.toPlainText() != '' and self.text_save.toPlainText() != '':
                self.button_detect.setEnabled(True)

    def read_result(self):
        self.vw.show()
        if self.pic.currentText() == '图片':
            path = self.par.save_path
            if len(self.file_list) == 0:
                for root, dirs, files in os.walk(path):
                    for file in files:
                        _, splitext = os.path.splitext(file)
                        if splitext == '.jpg' or splitext == '.png':
                            self.file_list.append(file)
                    if len(self.file_list) == 0:
                        QMessageBox.about(self, 'err', '图片或视频不存在')
            self.open_file = True
            file_name, splitext = os.path.splitext(self.file_list[self.i % len(self.file_list)])
            if splitext == '.jpg' or splitext == '.png':
                txt_path = path + '/labels/' + file_name + '.txt'
                if os.path.exists(txt_path):
                    self.label_is_smoke_.setText('Yes')
                    try:
                        txt = np.loadtxt(txt_path)
                        self.label_show_confidence_.setText(str(txt[5]))
                    except IndexError:
                        self.label_show_confidence_.setText('--')
                    self.label_start_time_.setText('--')
                    self.label_end_time_.setText('--')
                else:
                    self.label_is_smoke_.setText('No')
                    self.label_show_confidence_.setText('--')
                    self.label_start_time_.setText('--')
                    self.label_end_time_.setText('--')
            self.player.setMedia(QMediaContent(QUrl('file:///' + path + '/' + self.file_list[self.i % len(self.file_list)])))
            self.player.play()
        else:
            self.time_video.stop()
            list_txt = []
            if len(self.file_list) == 0:
                QMessageBox.about(self, 'err', '图片或视频不存在')
                return -1
            for root, dirs, files in os.walk(self.file_list[self.i % len(self.file_list)] + '/labels'):
                for file in files:
                    name, splitext = os.path.splitext(file)
                    if splitext == '.txt':
                        list_txt.append(int(name))
            self.open_file = True
            cap = cv2.VideoCapture(self.file_list[self.i % len(self.file_list)] + '/result.mp4')
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            if len(list_txt):
                list_txt.sort()
                self.label_start_time_.setText(str(list_txt[0] // self.fps) + 's')
                self.label_end_time_.setText(str(list_txt[-1] // self.fps) + 's')
            else:
                self.label_start_time_.setText('--')
                self.label_end_time_.setText('--')
            self.player.setMedia(QMediaContent(QUrl('file:///' + self.file_list[self.i % len(self.file_list)] + '/result.mp4')))
            self.player.play()
            self.time_video.start(int(1000 / self.fps))

    def confidence_of_video(self):
        duration = self.player.duration()
        position = self.player.position()
        if duration == position and duration != 0:
            self.time_video.stop()
        j = int(position * self.fps / 1000)
        path = self.file_list[self.i % len(self.file_list)] + '/labels/' + str(j - 1) + '.txt'
        if os.path.exists(path):
            txt = np.loadtxt(path)
            self.label_is_smoke_.setText('Yes')
            self.label_show_confidence_.setText(str(txt[5]))
        else:
            self.label_is_smoke_.setText('No')
            self.label_show_confidence_.setText('--')

    def i_plus(self):
        if self.open_file:
            self.i = self.i + 1
            if self.i >= len(self.file_list):
                self.i = self.i - len(self.file_list)
            self.read_result()

    def i_decrease(self):
        if self.open_file:
            self.i = self.i - 1
            if self.i <= -len(self.file_list):
                self.i = self.i + len(self.file_list)
            self.read_result()

    def real_time_detect(self):
        path = self.par.save_path + '/' + 'data'
        os.mkdir(path)
        path = self.par.save_path + '/' + 'detect'
        os.mkdir(path)
        self.model = DetectMultiBackend(self.par.model, device=torch.device('cuda'), dnn=False, data='', fp16=False)
        self.label_start_time_.setText('--')
        self.label_end_time_.setText('--')

    def show_realtime_img(self):
        self.j = self.j + 1
        _, self.realtime_img = self.cap.read()
        tim = time.strftime("%Y%m%d-%H%M%S")
        path = self.par.save_path + '/' + 'data'
        di = '/' + tim + '-' + str(self.j) + '.jpg'
        path = path + di
        cv2.imwrite(path, self.realtime_img)
        detect.run(weights=self.par.model, source=path, data='', imgsz=(640, 640),
                   conf_thres=self.par.confidence_threshold, iou_thres=self.par.NMS_threshold, max_det=self.par.max,
                   device='', view_img=False, save_txt=True, save_conf=True, save_crop=False, nosave=False,
                   classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, project='',
                   name='', exist_ok=False, line_thickness=3, hide_labels=self.par.hide_label, hide_conf=self.par.hide_confidence,
                   half=False, dnn=False, vid_stride=1, save_p=self.par.save_path + '/' + 'detect', model=self.model)
        path = self.par.save_path + '/' + 'detect' + di
        self.label_realtime_detect.setPixmap(QPixmap(path))
        path = self.par.save_path + '/detect/labels' + '/' + tim + '-' + str(self.j) + '.txt'
        if os.path.exists(path):
            self.label_is_smoke_.setText('Yes')
            txt = np.loadtxt(path)
            self.label_show_confidence_.setText(str(txt[5]))
        else:
            self.label_is_smoke_.setText('No')
            self.label_show_confidence_.setText('--')
        self.label_realtime_detect.setScaledContents(True)

    def run_detect(self):
        detect.run(weights=self.par.model, source=self.par.data_path, data='', imgsz=(640, 640),
                   conf_thres=self.par.confidence_threshold, iou_thres=self.par.NMS_threshold,
                   max_det=self.par.max,
                   device='', view_img=False, save_txt=True, save_conf=False, save_crop=False,
                   nosave=False,
                   classes=None, agnostic_nms=False, augment=False, visualize=False, update=False,
                   project='',
                   name='', exist_ok=False, line_thickness=3, hide_labels=self.par.hide_label,
                   hide_conf=self.par.hide_confidence, half=False,
                   dnn=False, vid_stride=1, save_p=self.par.save_path)
        self.file_list = []
        if self.result:
            self.read_result()
        self.label_message.setText('检测完毕')
        self.pic.setEnabled(True)
        if self.text_model.toPlainText() != '' and self.label_message.text() != '检测中' \
                and self.text_data.toPlainText() != '' and self.text_save.toPlainText() != '':
            self.button_detect.setEnabled(True)
        if self.is_realtime and self.text_model.toPlainText() != '' and self.text_save.toPlainText() != '':
            self.button_detect.setEnabled(True)

    def run_video_detect(self, path, file):
        cap = cv2.VideoCapture(self.par.data_path + '/' + file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps
        success, frame = cap.read()
        i = 0
        while success:
            cv2.imwrite(path + '/' + str(i) + '.jpg', frame)
            i = i + 1
            success, frame = cap.read()
        detect.run(weights=self.par.model, source=path, data='', imgsz=(640, 640),
                   conf_thres=self.par.confidence_threshold, iou_thres=self.par.NMS_threshold,
                   max_det=self.par.max,
                   device='', view_img=False, save_txt=True, save_conf=True, save_crop=False,
                   nosave=False,
                   classes=None, agnostic_nms=False, augment=False, visualize=False, update=False,
                   project='',
                   name='', exist_ok=False, line_thickness=3, hide_labels=self.par.hide_label,
                   hide_conf=self.par.hide_confidence, half=False, dnn=False, vid_stride=1, save_p=path)
        for root_, dirs_, files_ in os.walk(path):
            ls = []
            for file_ in files_:
                splitext = os.path.splitext(file_)
                if splitext[1] == '.jpg':
                    fname = os.path.join(root_, file_)
                    ls.append(fname)
            if len(ls):
                video_path = path + '/' + "result.mp4"
                if not os.path.exists(video_path):
                    size = cv2.imread(ls[0]).shape[1::-1]
                    videowriter = cv2.VideoWriter(video_path, -1, fps, size)
                    for x in ls:
                        img = cv2.imread(x)
                        videowriter.write(img)
                    videowriter.release()
                    self.file_list.append(path)
        if self.result:
            self.open_file = True
            self.read_result()
            self.time_video.start(int(1000 / self.fps))
            self.button_before.setEnabled(True)
            self.button_last.setEnabled(True)
            self.button_start.setEnabled(True)
            self.button_end.setEnabled(True)
        self.result = False

    def video_detect_finish(self, *lst):
        for item in lst:
            item.start()
        for item in lst:
            item.join()
        self.label_message.setText('检测完毕')
        self.pic.setEnabled(True)
        if self.text_model.toPlainText() != '' and self.label_message.text() != '检测中' \
                and self.text_data.toPlainText() != '' and self.text_save.toPlainText() != '':
            self.button_detect.setEnabled(True)
        if self.is_realtime and self.text_model.toPlainText() != '' and self.text_save.toPlainText() != '':
            self.button_detect.setEnabled(True)

    def start_detect(self):
        self.par.save_path = self.text_save.toPlainText()
        self.file_list = []
        if self.is_realtime:
            self.cap = cv2.VideoCapture(0)
            self.time_video.stop()
            self.button_realtime.setEnabled(False)
            tim = time.strftime("%Y%m%d-%H%M%S")
            self.label_detect_time_.setText(str(tim))
            self.par.save_path = self.par.save_path + '/' + tim
            os.mkdir(self.par.save_path)
            self.i = 0
            self.j = 0
            self.file_list = []
            self.par.data_path = 0
            self.par.model = self.text_model.toPlainText()
            self.par.confidence_threshold = float(self.lineEdit_confidence.text())
            self.par.NMS_threshold = float(self.lineEdit_NMS.text())
            self.par.max = int(self.lineEdit_max_target.text())
            if self.par.confidence_threshold > 1 or self.par.confidence_threshold <= 0:
                QMessageBox.about(self, 'error', '置信阈值应小于1大于0')
                self.lineEdit_confidence.setText('0.5')
                return -1
            if self.par.NMS_threshold > 1 or self.par.NMS_threshold <= 0:
                QMessageBox.about(self, 'error', 'NMS阈值应小于1大于0')
                self.lineEdit_NMS.setText('0.5')
                return -1
            if not os.path.exists(self.par.save_path):
                QMessageBox.about(self, 'error', '保存路径无效')
                self.text_save.setPlainText('')
                return -1
            if not os.path.isfile(self.par.model):
                QMessageBox.about(self, 'error', '模型无效')
                self.text_model.setPlainText('')
                return -1
            self.label_message.setText('检测中')
            self.label_message.repaint()
            self.real_time_detect()
            self.time.start(60)
            self.button_end.setEnabled(True)
            self.pic.setEnabled(False)
            self.button_detect.setEnabled(False)
            self.vw.hide()

        else:
            self.par.save_path = self.text_save.toPlainText()
            self.par.data_path = self.text_data.toPlainText()
            if self.pic.currentText() == '视频':
                self.button_start.setEnabled(False)
                self.button_end.setEnabled(False)
                self.button_before.setEnabled(False)
                self.button_last.setEnabled(False)
                for root, dirs, files in os.walk(self.par.data_path):
                    for file in files:
                        _, splitext = os.path.splitext(file)
                        if splitext == '.jpg' or splitext == '.png':
                            QMessageBox.about(self, 'error', '图片和视频不能一起检测')
                            self.text_data.setPlainText('')
                            return -1
                tim = time.strftime("%Y%m%d-%H%M%S")
                self.label_detect_time_.setText(str(tim))
                self.par.save_path = self.par.save_path + '/' + tim
                os.mkdir(self.par.save_path)
                self.i = 0
                self.file_list = []
                if not os.path.exists(self.par.data_path):
                    QMessageBox.about(self, 'error', '数据路径无效')
                    self.text_data.setPlainText('')
                    return -1
                self.par.model = self.text_model.toPlainText()
                self.par.confidence_threshold = float(self.lineEdit_confidence.text())
                self.par.NMS_threshold = float(self.lineEdit_NMS.text())
                self.par.max = int(self.lineEdit_max_target.text())
                if self.par.confidence_threshold > 1 or self.par.confidence_threshold <= 0:
                    QMessageBox.about(self, 'error', '置信阈值应小于1大于0')
                    self.lineEdit_confidence.setText('0.5')
                    return -1
                if self.par.NMS_threshold > 1 or self.par.NMS_threshold <= 0:
                    QMessageBox.about(self, 'error', 'NMS阈值应小于1大于0')
                    self.lineEdit_NMS.setText('0.5')
                    return -1
                if not os.path.exists(self.par.save_path):
                    QMessageBox.about(self, 'error', '保存路径无效')
                    self.text_save.setPlainText('')
                    return -1
                if not os.path.isfile(self.par.model):
                    QMessageBox.about(self, 'error', '模型无效')
                    self.text_model.setPlainText('')
                    return -1
                self.label_message.setText('检测中')
                self.pic.setEnabled(False)
                self.label_message.repaint()
                k = 0
                thr = []
                for root, dirs, files in os.walk(self.par.data_path):
                    for file in files:
                        k = k + 1
                        file_name, splitext = os.path.splitext(file)
                        if splitext == '.mp4':
                            path = self.par.save_path + '/' + file_name
                            os.mkdir(path)
                            thread_video_detect = Thread(target=self.run_video_detect, args=(path, file), name=str(k))
                            thr.append(thread_video_detect)
                thread_video_lst = Thread(target=self.video_detect_finish, args=thr, name='thr')
                thread_video_lst.start()
                button = QMessageBox.question(self, "Question", "是否打开检测结果",
                                              QMessageBox.Yes | QMessageBox.No,
                                              QMessageBox.Yes)
                if button == QMessageBox.Yes:
                    self.result = True
                if not self.is_realtime:
                    self.button_detect.setEnabled(False)
                self.text_data.setPlainText('')

            else:
                pic_num = 0
                for root, dirs, files in os.walk(self.par.data_path):
                    for file in files:
                        _, splitext = os.path.splitext(file)
                        if splitext == '.mp4':
                            QMessageBox.about(self, 'error', '图片和视频不能一起检测')
                            self.text_data.setPlainText('')
                            return -1
                        if splitext == '.jpg' or splitext == '.png':
                            pic_num = pic_num + 1
                if pic_num == 0:
                    QMessageBox.about(self, 'error', '文件夹中不存在待检测图片')
                    self.text_data.setPlainText('')
                    return -1
                tim = time.strftime("%Y%m%d-%H%M%S")
                self.label_detect_time_.setText(str(tim))
                self.par.save_path = self.par.save_path + '/' + tim
                os.mkdir(self.par.save_path)
                self.i = 0
                self.file_list = []
                if not os.path.exists(self.par.data_path):
                    QMessageBox.about(self, 'error', '数据路径无效')
                    self.text_data.setPlainText('')
                    return -1
                self.par.model = self.text_model.toPlainText()
                self.par.confidence_threshold = float(self.lineEdit_confidence.text())
                self.par.NMS_threshold = float(self.lineEdit_NMS.text())
                self.par.max = int(self.lineEdit_max_target.text())
                if self.par.confidence_threshold > 1 or self.par.confidence_threshold <= 0:
                    QMessageBox.about(self, 'error', '置信阈值应小于1大于0')
                    self.lineEdit_confidence.setText('0.5')
                    return -1
                if self.par.NMS_threshold > 1 or self.par.NMS_threshold <= 0:
                    QMessageBox.about(self, 'error', 'NMS阈值应小于1大于0')
                    self.lineEdit_NMS.setText('0.5')
                    return -1
                if not os.path.exists(self.par.save_path):
                    QMessageBox.about(self, 'error', '保存路径无效')
                    self.text_save.setPlainText('')
                    return -1
                if not os.path.isfile(self.par.model):
                    QMessageBox.about(self, 'error', '模型无效')
                    self.text_model.setPlainText('')
                    return -1
                self.label_message.setText('检测中')
                self.pic.setEnabled(False)
                self.label_message.repaint()
                thread_run = Thread(target=self.run_detect,
                                    name='run')
                thread_run.start()
                self.button_detect.setEnabled(False)
                wait = 1
                while wait:
                    time.sleep(2)
                    for root, dirs, files in os.walk(self.par.save_path):
                        for file in files:
                            _, splitext = os.path.splitext(file)
                            if splitext == '.jpg' or splitext == '.png':
                                wait = 0
                button = QMessageBox.question(self, "Question", "是否打开检测结果",
                                              QMessageBox.Yes | QMessageBox.No,
                                              QMessageBox.Yes)
                if button == QMessageBox.Yes:
                    self.read_result()
                    self.button_before.setEnabled(True)
                    self.button_last.setEnabled(True)
                if not self.is_realtime:
                    self.button_detect.setEnabled(False)
                self.text_data.setPlainText('')


def main():
    app = QApplication(sys.argv)
    ui = MainWindow()
    ui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
