import math
import random
import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtChart import QChart, QLineSeries, QValueAxis
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QFileDialog, QMainWindow
from interface import Ui_MainWindow


class PyQtMainEntry(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.OpenImage.clicked.connect(self.openImage)
        self.GrayingImg.clicked.connect(self.btnGray_Clicked)
        self.binaryzationImg.clicked.connect(self.btnBinaryzation_Clicked)

        self.fenxing.clicked.connect(self.btnfenxing_Clicked)
        self.GSI.clicked.connect(self.btnGSI_Clicked)
        self.Road.clicked.connect(self.btnRoad_Clicked)
        self.juzhen.clicked.connect(self.btnJuzhen_Clicked)
        self.juzhen_2.clicked.connect(self.btnJuzhen2_Clicked)
        self.out.clicked.connect(self.btnOut_Clicked)
        self.quzao.clicked.connect(self.quzao_Clicked)
        num = 72900

    def openImage(self):  # 选择本地图片上传
        filename, _ = QFileDialog.getOpenFileName(self, '打开图⽚')
        if filename:
            self.captured = cv2.imread(str(filename))
            self.captured = cv2.resize(self.captured, (270, 270), interpolation=cv2.INTER_AREA)
            # OpenCV图像以BGR通道存储，显⽰时需要从BGR转到RGB
            self.captured = cv2.cvtColor(self.captured, cv2.COLOR_BGR2RGB)

            rows, cols, channels = self.captured.shape
            bytesPerLine = channels * cols
            QImg = QImage(self.captured.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(QImg).scaled(
                self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnGray_Clicked(self):
        # 如果没有捕获图⽚，则不执⾏操作
        if not hasattr(self, "captured"):
            return
        self.cpatured = cv2.cvtColor(self.captured, cv2.COLOR_RGB2GRAY)

        rows, columns = self.cpatured.shape
        bytesPerLine = columns
        # 灰度图是单通道，所以需要⽤Format_Indexed8
        QImg = QImage(self.cpatured.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        self.label_2.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.label_2.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnBinaryzation_Clicked(self):
        # 如果没有捕获图片，则不执行操作
        if not hasattr(self, "captured"):
            return
        self.captured1 = cv2.GaussianBlur(self.captured, (3, 3), 0)
        self.gray = cv2.cvtColor(self.captured1, cv2.COLOR_RGB2GRAY)
        self.binary = cv2.adaptiveThreshold(self.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 15)

        rows, columns = self.binary.shape
        bytesPerLine = columns
        # 灰度图是单通道，所以需要⽤Format_Indexed8
        QImg = QImage(self.binary.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        self.label_3.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.label_3.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def quzao_Clicked(self):

        self.binary = 255 - self.binary

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self.binary, connectivity=8)
        print(num_labels)
        areas = list()
        for i in range(num_labels):
            areas.append(stats[i][-1])
            # print("轮廓%d的面积:%d" % (i, stats[i][-1]))
        k = 3
        area_avg = np.average(areas[1:-1]) * k
        print("轮廓平均面积:", '%.2f' % area_avg)

        d = z = z1 = 0
        image_filtered = np.zeros_like(self.binary)
        for (i, label) in enumerate(np.unique(labels)):
            # 如果是背景，忽略
            if label == 0:
                continue
            if stats[i][-1] > area_avg:
                image_filtered[labels == i] = 255
                d = d + 1  # 数量
                n = stats[i][2] * stats[i][2] + stats[i][3] * stats[i][3]  # 勾股定理
                x = math.sqrt(n) * 1.1  # 每条长度 1.1为系数
                z = z + x  # 累加总长度
                y = stats[i][-1] / x  # 宽度
                z1 = z1 + y  # 累加总宽度

        print("数量:", d)
        print("平均长度:", '%.2f' % (z / d))
        print("平均宽度:", '%.2f' % (z1 / d))
        image_filtered = 255 - image_filtered
        p2 = 0
        for k in range(0, 270):
            for n in range(0, 270):
                if image_filtered[k, n].all() == 0:
                    p2 += 1
        print("占有率:", '%.2f' % (p2 / 270 / 270))

        rows, columns = image_filtered.shape
        bytesPerLine = columns
        # 灰度图是单通道，所以需要⽤Format_Indexed8
        QImg = QImage(image_filtered.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        self.label_3.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.label_3.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self.binary = image_filtered

    def btnfenxing_Clicked(self):
        p2 = p3 = 1
        for m in range(0, 270, 30):
            for n in range(0, 270, 30):
                for x in range(m, m + 30):
                    for y in range(n, n + 30):
                        px2 = int(self.binary[x, y])
                        if px2 == 0:
                            p2 += 1
                k2 = 99 - (int((p2 / 900) * 450))
                if k2 < 0:
                    k2 = k2 * (-1)
                print(k2)
                f = int(m / 30)
                h = int(n / 30)
                print(f, h)
                a[f][h] = k2
                p2 = 0

        # 如果没有捕获图⽚，则不执⾏操作
        if not hasattr(self, "binary"):
            return
        self.captured2 = cv2.cvtColor(self.binary, cv2.COLOR_BGR2RGB)
        #self.captured2 = 255 - self.captured2
        self.fxws = cv2.line(self.captured2, (269, 0), (269, 270), (255, 0, 0), 2)  # 参数分别是起始点，重点，颜色：蓝色，厚度：5 px
        self.fxws = cv2.line(self.captured2, (0, 269), (269, 269), (255, 0, 0), 2)  # 参数分别是起始点，重点，颜色：蓝色，厚度：5 px
        for i in range(0, 271, 30):
            self.fxws = cv2.line(self.captured2, (i, 0), (i, 270), (255, 0, 0), 2)  # 参数分别是起始点，重点，颜色：蓝色，厚度：5 px
        for m in range(0, 271, 30):
            self.fxws = cv2.line(self.captured2, (0, m), (270, m), (255, 0, 0), 2)  # 参数分别是起始点，重点，颜色：蓝色，厚度：5 px
        for x in range(0, 9, 1):
            for y in range(0, 9, 1):
                font = cv2.FONT_HERSHEY_PLAIN
                cv2.putText(self.fxws, str(a[y][x]), (int(x) * 30 + 5, int(y) * 30 + 25), font, 1, (255, 0, 0), 1,
                            cv2.LINE_AA)

        rows, cols, channels = self.fxws.shape
        bytesPerLine = channels * cols
        QImg = QImage(self.fxws.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
        self.label_4.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.label_4.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnGSI_Clicked(self):
        # 如果没有捕获图⽚，则不执⾏操作
        if not hasattr(self, "binary"):
            return
        self.img5 = np.ones((270, 270, 3), np.uint8) * 255  # 生成一个图像

        self.fxws = cv2.line(self.img5, (269, 0), (269, 270), (255, 0, 0), 1)  # 参数分别是起始点，重点，颜色：蓝色，厚度：5 px
        self.fxws = cv2.line(self.img5, (0, 269), (269, 269), (255, 0, 0), 1)  # 参数分别是起始点，重点，颜色：蓝色，厚度：5 px
        for i in range(0, 271, 30):
            self.fxws = cv2.line(self.img5, (i, 0), (i, 270), (255, 0, 0), 1)
        for m in range(0, 271, 30):
            self.fxws = cv2.line(self.img5, (0, m), (270, m), (255, 0, 0), 1)
        for x in range(0, 9, 1):
            for y in range(0, 9, 1):
                font = cv2.FONT_HERSHEY_PLAIN
                cv2.putText(self.img5, str(a[y][x]), (int(x) * 30 + 5, int(y) * 30 + 25), font, 1, (255, 0, 0), 1,
                            cv2.LINE_AA)

            rows, cols, channels = self.fxws.shape
            bytesPerLine = channels * cols
            QImg = QImage(self.fxws.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
            self.label_5.setPixmap(QPixmap.fromImage(QImg).scaled(
                self.label_5.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnRoad_Clicked(self):
        self.Img = cv2.imread('d3.png')  # 通过opencv读入一张图片
        self.Img = cv2.cvtColor(self.Img, cv2.COLOR_BGR2RGB)
        rows, cols, channels = self.Img.shape
        bytesPerLine = channels * cols
        QImg = QImage(self.Img.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
        self.label_6.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.label_6.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnJuzhen_Clicked(self):
        self.label_7.setText(str(a))

    def btnJuzhen2_Clicked(self):

        self.label_8.setText("截割路径沿线地质强度指标为：" +
                             str(a[7][1]) + "→" + str(a[7][2]) + "→" + str(a[7][3]) + "→" + str(a[7][4]) + "→" + str(
            a[7][5]) + "→" + str(a[7][6]) + "→" + str(a[7][7]) + " "
                             + "→" + str(a[6][7]) + "→" + str(a[5][7]) + "→" + str(a[5][6]) + "→" + str(
            a[5][5]) + "→" + str(a[5][4]) + "→" + str(a[5][3]) + "→" + str(a[5][2]) + " " + "→" + str(a[5][1])
                             + "→" + str(a[4][1]) + "→" + str(a[3][1]) + "→" + str(a[3][2]) + "→" + str(
            a[3][3]) + "→" + str(a[3][4]) + "→" + str(a[3][5]) + " " + "→" + str(a[3][6]) + "→" + str(a[3][7])
                             + "→" + str(a[2][7]) + "→" + str(a[1][7]) + "→" + str(a[1][6]) + "→" + str(
            a[1][5]) + "→" + str(a[1][4]) + " " + "→" + str(a[1][3]) + "→" + str(a[1][2]) + "→" + str(a[1][1]))

    def btnOut_Clicked(self):
        chart = QChart()
        chart.setTitle("截割沿线GSI")
        self.graphicsView.setChart(chart)
        seri = QLineSeries()
        seri.setName("地质强度指标GSI")
        chart.addSeries(seri)

        self.array_a = [a[7][1], a[7][2], a[7][3], a[7][4], a[7][5], a[7][6], a[7][7], a[6][7], a[5][7], a[5][6],
                        a[5][5], a[5][4], a[5][3], a[5][2],
                        a[5][1], a[4][1], a[3][1], a[3][2], a[3][3], a[3][4], a[3][5], a[3][6], a[3][7]
            , a[2][7], a[1][7], a[1][6], a[1][5], a[1][4], a[1][3], a[1][2], a[1][1]]
        for i in range(31):
            seri.append(i, self.array_a[i])

        ax = QValueAxis()
        ax.setRange(0, 30)
        ax.setTitleText("截割路径")
        ax.setLabelFormat("%d")
        ay = QValueAxis()
        ay.setRange(0, 100)
        ay.setTitleText("GSI")
        ay.setLabelFormat("%d")
        chart.setAxisX(ax, seri)
        chart.setAxisY(ay, seri)

        chart1 = QChart()
        chart1.setTitle("截割频率")
        self.graphicsView_2.setChart(chart1)
        seri1 = QLineSeries()
        seri1.setName("截割频率(Hz)")
        chart1.addSeries(seri1)

        self.array_b = [a[7][1], a[7][2], a[7][3], a[7][4], a[7][5], a[7][6], a[7][7], a[6][7], a[5][7], a[5][6],
                        a[5][5], a[5][4], a[5][3], a[5][2],
                        a[5][1], a[4][1], a[3][1], a[3][2], a[3][3], a[3][4], a[3][5], a[3][6], a[3][7]
            , a[2][7], a[1][7], a[1][6], a[1][5], a[1][4], a[1][3], a[1][2], a[1][1]]

        for i in range(31):
            if self.array_b[i] >= 70:
                self.array_b[i] = 50
            elif self.array_b[i] <= 35:
                self.array_b[i] = 30
            else:
                self.array_b[i] = 40
            seri1.append(i * 10, self.array_b[i])

        ax1 = QValueAxis()
        ax1.setRange(0, 300)
        ax1.setLabelFormat("%d")
        ax1.setTitleText("运行时间(s)")
        ay1 = QValueAxis()
        ay1.setRange(20, 60)
        ay1.setTitleText("频率(Hz)")
        ay1.setLabelFormat("%d")
        chart1.setAxisX(ax1, seri1)
        chart1.setAxisY(ay1, seri1)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    a = np.random.randint(10, 99, (9, 9))
    #a[][]=[0][0]
    window = PyQtMainEntry()
    window.show()
    sys.exit(app.exec_())
