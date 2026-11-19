from PyQt5 import QtWidgets, uic, QtGui, QtCore
import sys
import cv2
from forensics import (
    hill_climbing, iterative, scale_image, find_forgeries, draw_disp_arrows
)


class State:
    NotImported = 1
    Imported = 2
    FoundGlobals = 3
    FoundForgeries = 4


class WorkerThread(QtCore.QThread):
    finished = QtCore.pyqtSignal(tuple)
    progress = QtCore.pyqtSignal(float)

    def __init__(self, target, *args):
        QtCore.QThread.__init__(self)
        self.target = target
        self.args = args

    def run(self):
        result = self.target(*self.args)
        self.finished.emit(result)

    def stop(self):
        self.terminate()


def cvImageToQImage(img):
    height, width, channel = img.shape
    bytesPerLine = 3 * width
    return QtGui.QImage(
        img,
        width,
        height,
        bytesPerLine,
        QtGui.QImage.Format_RGB888).rgbSwapped()


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        # Call the inherited classes __init__ method
        super(Ui, self).__init__()
        uic.loadUi('ui.ui', self)  # Load the .ui file
        self.setFixedSize(1344, 662)

        # Widgets
        self.outputImg = self.findChild(QtWidgets.QLabel, 'outputPic')
        self.inputImg = self.findChild(QtWidgets.QLabel, 'inputPic')
        self.start = self.findChild(QtWidgets.QPushButton, 'start')
        self.abort = self.findChild(QtWidgets.QPushButton, 'abort')
        self.save = self.findChild(QtWidgets.QPushButton, 'save')
        self.fileselect = self.findChild(QtWidgets.QPushButton, 'load')
        self.displayCombo = self.findChild(QtWidgets.QComboBox, 'displayCombo')
        self.inputCombo = self.findChild(QtWidgets.QComboBox, 'inputCombo')
        self.progress = self.findChild(QtWidgets.QProgressBar, 'progressBar')
        self.methodLabel = self.findChild(QtWidgets.QLabel, 'methodLabel')
        self.methodCombo = self.findChild(QtWidgets.QComboBox, 'methodCombo')
        self.roiWidthLabel = self.findChild(QtWidgets.QLabel, 'roiWidthLabel')
        self.roiHeightLabel = self.findChild(
            QtWidgets.QLabel, 'roiHeightLabel')
        self.roiWidth = self.findChild(QtWidgets.QSpinBox, 'roiWidth')
        self.roiHeight = self.findChild(QtWidgets.QSpinBox, 'roiHeight')
        self.errorLimitLabel = self.findChild(QtWidgets.QLabel, 'errorLimitLabel')
        self.errorLimit = self.findChild(QtWidgets.QDoubleSpinBox, 'errorLimit')

        self.state = State.NotImported
        self.inputImgCv = None
        self.outputImgCv = None
        self.correctedImg = None
        self.blueDispImg = None
        self.redDispImg = None
        self.forgeriesImg = None
        self.roiBlueDispImg = None
        self.roiRedDispImg = None
        self.workerThread = None
        self.method = hill_climbing
        self.alpha_blue = 1
        self.alpha_red = 1

        # Connections
        self.fileselect.clicked.connect(self.loadPressed)
        self.start.clicked.connect(self.startPressed)
        self.abort.clicked.connect(self.abortPressed)
        self.save.clicked.connect(self.savePressed)
        self.inputCombo.currentIndexChanged.connect(self.inputChanged)
        self.displayCombo.currentIndexChanged.connect(self.displayChanged)
        self.methodCombo.currentIndexChanged.connect(self.methodChanged)

        self.setState(State.NotImported)
        self.show()  # Show the GUI

    def loadPressed(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file')
        if filename:
            self.inputImgCv = cv2.imread(filename)
            height, width, channel = self.inputImgCv.shape
            bytesPerLine = 3 * width
            qImg = QtGui.QImage(
                self.inputImgCv,
                width,
                height,
                bytesPerLine,
                QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap(qImg)
            pix = pix.scaled(500, 500, QtCore.Qt.KeepAspectRatio)
            self.inputImg.setPixmap(pix)
            self.setState(State.Imported)

    def savePressed(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save file', '', 'All files (*.*);;PNG (*.png)',
            'PNG (*.png)')
        if filename:
            cv2.imwrite(filename, self.outputImgCv)

    def startPressed(self):
        if self.inputCombo.currentIndex() == 0:
            self.runFindGlobalParams()
        else:
            self.runFindForgeries()
        self.progress.setValue(0)
        self.inputCombo.setEnabled(False)
        self.methodCombo.setEnabled(False)

    def abortPressed(self):
        if self.workerThread is not None:
            self.workerThread.stop()
            self.workerThread = None
        self.inputRunFinished()

    def inputChanged(self, index):
        if index == 0:
            self.roiWidthLabel.hide()
            self.roiHeightLabel.hide()
            self.errorLimitLabel.hide()
            self.roiWidth.hide()
            self.roiHeight.hide()
            self.errorLimit.hide()
        else:
            self.roiWidthLabel.show()
            self.roiHeightLabel.show()
            self.errorLimitLabel.show()
            self.roiWidth.show()
            self.roiHeight.show()
            self.errorLimit.show()

    def displayChanged(self, index):
        if index == 0:
            self.outputImgCv = self.correctedImg
        if index == 1:
            self.outputImgCv = self.blueDispImg
        if index == 2:
            self.outputImgCv = self.redDispImg
        if index == 3:
            self.outputImgCv = self.forgeriesImg
        if index == 4:
            self.outputImgCv = self.roiBlueDispImg
        if index == 5:
            self.outputImgCv = self.roiRedDispImg
        self.updateDisplayImage()

    def methodChanged(self, index):
        if index == 0:
            self.method = hill_climbing
        else:
            self.method = iterative

    def updateDisplayImage(self):
        qImg = cvImageToQImage(self.outputImgCv)
        pix = QtGui.QPixmap(qImg)
        pix = pix.scaled(500, 500, QtCore.Qt.KeepAspectRatio)
        self.outputImg.setPixmap(pix)

    def newGlobalParams(self, result):
        center, alpha_blue, alpha_red = result
        self.alpha_blue = alpha_blue
        self.alpha_red = alpha_red
        image = self.inputImgCv

        # Correct blue and red channels
        corr_blue = scale_image(image[:, :, 0], 1 / alpha_blue, center, (0, 0))
        corr_red = scale_image(image[:, :, 2], 1 / alpha_red, center, (0, 0))
        self.correctedImg = cv2.merge((corr_blue, image[:, :, 1], corr_red))
        self.outputImgCv = self.correctedImg

        self.blueDispImg, self.redDispImg = draw_disp_arrows(
            self.inputImgCv, center, alpha_blue, alpha_red)

        self.updateDisplayImage()
        self.setState(State.FoundGlobals)

        # Reset functionality
        self.workerThread = None
        self.inputRunFinished()

    def newForgeries(self, result):
        self.forgeriesImg, _ = result
        self.outputImgCv = self.forgeriesImg

        self.updateDisplayImage()
        self.setState(State.FoundForgeries)

        # Reset functionality
        self.workerThread = None
        self.inputRunFinished()

    def inputRunStarted(self):
        self.fileselect.setEnabled(False)
        self.start.setEnabled(False)
        self.abort.setEnabled(True)
        self.progress.setValue(0)
        self.inputCombo.setEnabled(False)
        self.methodCombo.setEnabled(False)

    def inputRunFinished(self):
        self.fileselect.setEnabled(True)
        self.start.setEnabled(True)
        self.abort.setEnabled(False)
        self.progress.setValue(0)
        self.inputCombo.setEnabled(True)
        self.methodCombo.setEnabled(True)

    def updateProgress(self, value):
        self.progress.setValue(int(value * 100))

    def runFindGlobalParams(self):
        height, width, _ = self.inputImgCv.shape
        self.workerThread = WorkerThread(
            self.method,
            self.inputImgCv, (0, 0), (height, width),
            lambda x: self.workerThread.progress.emit(x))
        self.workerThread.finished.connect(self.newGlobalParams)
        self.workerThread.progress.connect(self.updateProgress)

        self.inputRunStarted()
        self.workerThread.start()

    def runFindForgeries(self):
        roi_height = self.roiHeight.value()
        roi_width = self.roiWidth.value()
        err_limit = self.errorLimit.value()
        self.workerThread = WorkerThread(
            find_forgeries,
            self.inputImgCv, self.method, (0, 0),
            self.alpha_blue, self.alpha_red,
            roi_height, roi_width, err_limit, False,
            lambda x: self.workerThread.progress.emit(x))
        self.workerThread.finished.connect(self.newForgeries)
        self.workerThread.progress.connect(self.updateProgress)

        self.inputRunStarted()
        self.workerThread.start()

    def setState(self, state):
        self.state = state
        inputModel = self.inputCombo.model()
        displayModel = self.displayCombo.model()
        self.inputCombo.setEnabled(False)
        self.displayCombo.setEnabled(False)
        self.methodLabel.hide()
        self.methodCombo.hide()
        self.roiWidthLabel.hide()
        self.roiHeightLabel.hide()
        self.errorLimitLabel.hide()
        self.roiWidth.hide()
        self.roiHeight.hide()
        self.errorLimit.hide()

        if state >= State.Imported:
            self.start.setEnabled(True)
            self.inputCombo.setEnabled(True)
            inputModel.item(0).setEnabled(True)
            self.methodLabel.show()
            self.methodCombo.show()

            if state < State.FoundGlobals:
                inputModel.item(1).setEnabled(False)
                self.save.setEnabled(False)
                self.inputCombo.setCurrentIndex(0)
                self.inputChanged(0)
                # Set min/max and default values on ROIs
                height, width, _ = self.inputImgCv.shape
                self.roiWidth.setRange(1, width)
                self.roiHeight.setRange(1, height)
                self.roiWidth.setValue(width // 4)
                self.roiHeight.setValue(height // 4)

        if state >= State.FoundGlobals:
            self.save.setEnabled(True)
            inputModel.item(1).setEnabled(True)
            self.displayCombo.setEnabled(True)
            displayModel.item(0).setEnabled(True)
            displayModel.item(1).setEnabled(True)
            displayModel.item(2).setEnabled(True)

            if state < State.FoundForgeries:
                displayModel.item(3).setEnabled(False)
                self.displayCombo.setCurrentIndex(0)
                self.displayChanged(0)

        if state == State.FoundForgeries:
            displayModel.item(3).setEnabled(True)
            self.displayCombo.setCurrentIndex(3)
            self.displayChanged(3)


app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()
