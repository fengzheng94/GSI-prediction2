# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Interface.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1083, 647)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(40, 130, 200, 200))
        self.label.setStyleSheet("background-color:rgb(255, 255, 255)")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(260, 130, 200, 200))
        self.label_2.setStyleSheet("background-color:rgb(255, 255, 255)")
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(480, 130, 200, 200))
        self.label_3.setStyleSheet("background-color:rgb(255, 255, 255)")
        self.label_3.setObjectName("label_3")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(670, 400, 113, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lineEdit.setFont(font)
        self.lineEdit.setText("")
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(670, 440, 113, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lineEdit_2.setFont(font)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_3.setGeometry(QtCore.QRect(670, 480, 113, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lineEdit_3.setFont(font)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.lineEdit_4 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_4.setGeometry(QtCore.QRect(670, 520, 113, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lineEdit_4.setFont(font)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(940, 470, 71, 71))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.Result_label = QtWidgets.QLabel(self.centralwidget)
        self.Result_label.setGeometry(QtCore.QRect(970, 410, 71, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.Result_label.setFont(font)
        self.Result_label.setStyleSheet("background-color:rgb(255, 255, 255);color:rgb(170, 0, 0)")
        self.Result_label.setText("")
        self.Result_label.setAlignment(QtCore.Qt.AlignCenter)
        self.Result_label.setObjectName("Result_label")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(900, 410, 71, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_5.setFont(font)
        self.label_5.setStyleSheet("background-color:rgb(255, 255, 255)")
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(603, 400, 61, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_6.setFont(font)
        self.label_6.setStyleSheet("background-color:rgb(255, 255, 255)")
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(603, 440, 61, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_7.setFont(font)
        self.label_7.setStyleSheet("background-color:rgb(255, 255, 255)")
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(603, 480, 61, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_8.setFont(font)
        self.label_8.setStyleSheet("background-color:rgb(255, 255, 255)")
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(603, 520, 61, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_9.setFont(font)
        self.label_9.setStyleSheet("background-color:rgb(255, 255, 255)")
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(790, 400, 54, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_10.setFont(font)
        self.label_10.setStyleSheet("background-color:rgb(255, 255, 255)")
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(790, 440, 54, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_11.setFont(font)
        self.label_11.setStyleSheet("background-color:rgb(255, 255, 255)")
        self.label_11.setAlignment(QtCore.Qt.AlignCenter)
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(790, 480, 54, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_12.setFont(font)
        self.label_12.setStyleSheet("background-color:rgb(255, 255, 255)")
        self.label_12.setAlignment(QtCore.Qt.AlignCenter)
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(790, 520, 54, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_13.setFont(font)
        self.label_13.setStyleSheet("background-color:rgb(255, 255, 255)")
        self.label_13.setAlignment(QtCore.Qt.AlignCenter)
        self.label_13.setObjectName("label_13")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(0, 0, 1081, 81))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(24)
        self.label_4.setFont(font)
        self.label_4.setStyleSheet("background-color:rgb(0, 85, 255);color:rgb(255, 255, 255)")
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        self.label_14.setGeometry(QtCore.QRect(700, 130, 200, 200))
        self.label_14.setStyleSheet("background-color:rgb(255, 255, 255)")
        self.label_14.setObjectName("label_14")
        self.label_15 = QtWidgets.QLabel(self.centralwidget)
        self.label_15.setGeometry(QtCore.QRect(610, 570, 81, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_15.setFont(font)
        self.label_15.setStyleSheet("background-color:rgb(255, 255, 255)")
        self.label_15.setObjectName("label_15")
        self.erzhi = QtWidgets.QPushButton(self.centralwidget)
        self.erzhi.setGeometry(QtCore.QRect(930, 210, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.erzhi.setFont(font)
        self.erzhi.setObjectName("erzhi")
        self.huidu = QtWidgets.QPushButton(self.centralwidget)
        self.huidu.setGeometry(QtCore.QRect(930, 170, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.huidu.setFont(font)
        self.huidu.setObjectName("huidu")
        self.quzao = QtWidgets.QPushButton(self.centralwidget)
        self.quzao.setGeometry(QtCore.QRect(930, 250, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.quzao.setFont(font)
        self.quzao.setObjectName("quzao")
        self.open = QtWidgets.QPushButton(self.centralwidget)
        self.open.setGeometry(QtCore.QRect(930, 130, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.open.setFont(font)
        self.open.setObjectName("open")
        self.label_16 = QtWidgets.QLabel(self.centralwidget)
        self.label_16.setGeometry(QtCore.QRect(0, 80, 1081, 541))
        self.label_16.setStyleSheet("background-color:rgb(0, 85, 127)")
        self.label_16.setText("")
        self.label_16.setObjectName("label_16")
        self.jisuan = QtWidgets.QPushButton(self.centralwidget)
        self.jisuan.setGeometry(QtCore.QRect(930, 290, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.jisuan.setFont(font)
        self.jisuan.setObjectName("jisuan")
        self.label_17 = QtWidgets.QLabel(self.centralwidget)
        self.label_17.setGeometry(QtCore.QRect(580, 350, 491, 251))
        self.label_17.setStyleSheet("border: 2px dashed white")
        self.label_17.setText("")
        self.label_17.setObjectName("label_17")
        self.label_18 = QtWidgets.QLabel(self.centralwidget)
        self.label_18.setGeometry(QtCore.QRect(40, 390, 200, 200))
        self.label_18.setStyleSheet("background-color:rgb(255, 255, 255)")
        self.label_18.setObjectName("label_18")
        self.label_19 = QtWidgets.QLabel(self.centralwidget)
        self.label_19.setGeometry(QtCore.QRect(260, 390, 200, 200))
        self.label_19.setStyleSheet("background-color:rgb(255, 255, 255)")
        self.label_19.setObjectName("label_19")
        self.label_20 = QtWidgets.QLabel(self.centralwidget)
        self.label_20.setGeometry(QtCore.QRect(10, 90, 1061, 251))
        self.label_20.setStyleSheet("border: 2px dashed white")
        self.label_20.setText("")
        self.label_20.setObjectName("label_20")
        self.label_21 = QtWidgets.QLabel(self.centralwidget)
        self.label_21.setGeometry(QtCore.QRect(40, 100, 181, 18))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(12)
        self.label_21.setFont(font)
        self.label_21.setStyleSheet("color:rgb(255, 255, 255)")
        self.label_21.setObjectName("label_21")
        self.label_22 = QtWidgets.QLabel(self.centralwidget)
        self.label_22.setGeometry(QtCore.QRect(10, 350, 561, 251))
        self.label_22.setStyleSheet("border: 2px dashed white")
        self.label_22.setText("")
        self.label_22.setObjectName("label_22")
        self.label_23 = QtWidgets.QLabel(self.centralwidget)
        self.label_23.setGeometry(QtCore.QRect(40, 360, 181, 18))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(12)
        self.label_23.setFont(font)
        self.label_23.setStyleSheet("color:rgb(255, 255, 255)")
        self.label_23.setObjectName("label_23")
        self.label_24 = QtWidgets.QLabel(self.centralwidget)
        self.label_24.setGeometry(QtCore.QRect(600, 360, 221, 18))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(12)
        self.label_24.setFont(font)
        self.label_24.setStyleSheet("color:rgb(255, 255, 255)")
        self.label_24.setObjectName("label_24")
        self.open_2 = QtWidgets.QPushButton(self.centralwidget)
        self.open_2.setGeometry(QtCore.QRect(480, 390, 81, 81))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.open_2.setFont(font)
        self.open_2.setObjectName("open_2")
        self.jisuan_2 = QtWidgets.QPushButton(self.centralwidget)
        self.jisuan_2.setGeometry(QtCore.QRect(480, 500, 81, 81))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.jisuan_2.setFont(font)
        self.jisuan_2.setObjectName("jisuan_2")
        self.label_16.raise_()
        self.label_22.raise_()
        self.label_20.raise_()
        self.label_17.raise_()
        self.label.raise_()
        self.label_2.raise_()
        self.label_3.raise_()
        self.lineEdit.raise_()
        self.lineEdit_2.raise_()
        self.lineEdit_3.raise_()
        self.lineEdit_4.raise_()
        self.pushButton.raise_()
        self.Result_label.raise_()
        self.label_5.raise_()
        self.label_6.raise_()
        self.label_7.raise_()
        self.label_8.raise_()
        self.label_9.raise_()
        self.label_10.raise_()
        self.label_11.raise_()
        self.label_12.raise_()
        self.label_13.raise_()
        self.label_4.raise_()
        self.label_14.raise_()
        self.label_15.raise_()
        self.erzhi.raise_()
        self.huidu.raise_()
        self.quzao.raise_()
        self.open.raise_()
        self.jisuan.raise_()
        self.label_18.raise_()
        self.label_19.raise_()
        self.label_21.raise_()
        self.label_23.raise_()
        self.label_24.raise_()
        self.open_2.raise_()
        self.jisuan_2.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "原图"))
        self.label_2.setText(_translate("MainWindow", "灰度图"))
        self.label_3.setText(_translate("MainWindow", "二值图"))
        self.pushButton.setText(_translate("MainWindow", "预测"))
        self.label_5.setText(_translate("MainWindow", "GSI:"))
        self.label_6.setText(_translate("MainWindow", "数量："))
        self.label_7.setText(_translate("MainWindow", "长度："))
        self.label_8.setText(_translate("MainWindow", "宽度："))
        self.label_9.setText(_translate("MainWindow", "密度："))
        self.label_10.setText(_translate("MainWindow", "条"))
        self.label_11.setText(_translate("MainWindow", "mm"))
        self.label_12.setText(_translate("MainWindow", "mm"))
        self.label_13.setText(_translate("MainWindow", "%"))
        self.label_4.setText(_translate("MainWindow", "岩石裂隙节理图像检测及其地质强度指标检测系统"))
        self.label_14.setText(_translate("MainWindow", "裂隙图"))
        self.label_15.setText(_translate("MainWindow", "TextLabel"))
        self.erzhi.setText(_translate("MainWindow", "二值化"))
        self.huidu.setText(_translate("MainWindow", "灰度"))
        self.quzao.setText(_translate("MainWindow", "去噪"))
        self.open.setText(_translate("MainWindow", "打开图像"))
        self.jisuan.setText(_translate("MainWindow", "计算"))
        self.label_18.setText(_translate("MainWindow", "原图"))
        self.label_19.setText(_translate("MainWindow", "语义分割结果"))
        self.label_21.setText(_translate("MainWindow", "● 传统图像检测方法"))
        self.label_23.setText(_translate("MainWindow", "● U-Net语义分割方法"))
        self.label_24.setText(_translate("MainWindow", "◆ 地质强度指标GSI预测"))
        self.open_2.setText(_translate("MainWindow", "打开图像"))
        self.jisuan_2.setText(_translate("MainWindow", "计算"))
