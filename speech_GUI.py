# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 19:38:53 2020

@author: Ivan Nemov
"""
from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib.pyplot as plt
import pyaudio
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg
import speech
import os
import csv

DataSelBoxWidth = 350
SpeechControlBoxWidth = 350
silence_treshold_slider = 0
stft_segment_size = 900
command_duration = 0.6
command_duration_manual = 0.6
command_repeats = 1
FREQ = 20
RATE = 48000
CHUNK = RATE/FREQ
COMMAND_CHUNK = CHUNK
stft_freq_res = 20
stft_freq_high_lim = 10000
stft_overlap_percent = 90
stft_lin_log_norm = 1
audio_recording_output_format = 1
personal_data_feed_slider = 0
CNN_batches = 30
CNN_epochs = 20
cnn_silence_treshold_slider = 0

class SpeechRecognitionWindow(QtWidgets.QMainWindow):

    def __init__(self, _quit_signal_message, parent=None):
        
        self._quit_signal_message = _quit_signal_message
        
        super(SpeechRecognitionWindow, self).__init__(parent)
        self.form_widget = FormWidget(self, self._quit_signal_message)
        _widget = QtWidgets.QWidget()
        _layout = QtWidgets.QVBoxLayout(_widget)
        _layout.addWidget(self.form_widget)
        self.setCentralWidget(_widget)
        self.resize(1000, 515)
        self.setWindowTitle("Skeet Shooting Digital Assistant: configuration of speech recognition")
        self.setWindowIcon(QtGui.QIcon(QtCore.QDir.currentPath()+'/GUI_images/speech_recognition.png'))
        self.quit = QtWidgets.QAction("Quit", self)
        self.quit.triggered.connect(self.closeEvent)
        
    def closeEvent(self, event):
        self.form_widget.exit_action_custom()
        event.ignore()
        
class FormWidget(QtWidgets.QWidget):
    
    def __init__(self, parent, _quit_signal_message):
        super(FormWidget, self).__init__(parent)
        self.__controls()
        self.__layout()
        self._quit_signal_message = _quit_signal_message

    def __controls(self):
        
        global DataSelBoxWidth
        global SpeechControlBoxWidth
        global silence_treshold_slider
        global stft_segment_size
        global command_duration
        global FREQ
        global RATE
        global CHUNK
        global COMMAND_CHUNK
        global command_repeats
        global stft_freq_res
        global stft_freq_high_lim
        global stft_overlap_percent
        global stft_lin_log_norm
        global personal_data_feed_slider
        global CNN_batches
        global CNN_epochs
        global cnn_silence_treshold_slider
        
        self.menu_bar=QtWidgets.QMenuBar()
        self.menu_bar.setFixedHeight(20)
        file_menu=self.menu_bar.addMenu("File")
        exit_action=QtWidgets.QAction('Exit',self)
        exit_action.triggered.connect(self.exit_action_custom)
        file_menu.addAction(exit_action)
        
        #File tree
        self.DataSelBox = QtWidgets.QGroupBox(self)
        self.DataSelBox.setObjectName("DataSelBox")
        self.DataSelBox.setFixedWidth(DataSelBoxWidth)
        self.DataSelBox.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        self.DataSelBox.setTitle("Data selection")
        
        self.treeView = QtWidgets.QTreeView(self.DataSelBox)
        self.treeView.setObjectName("treeView")
        self.treemodel=QtWidgets.QFileSystemModel()
        rootpath=QtCore.QDir.currentPath()+'/speech'
        self.treemodel.setRootPath(rootpath)
        self.treeView.setModel(self.treemodel)
        self.treeView.setRootIndex(self.treemodel.index(rootpath))
        self.treeView.setColumnWidth(0, 800)
        
        self.AddPlotButton = QtWidgets.QPushButton(self.DataSelBox)
        self.AddPlotButton.setObjectName("AddPlotButton")
        self.AddPlotButton.setFixedWidth(75)
        self.AddPlotButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.AddPlotButton.setText("Plot STFT")
        self.AddPlotButton.clicked.connect(self.plot_stft)
        
        self.DeleteFileButton = QtWidgets.QPushButton(self.DataSelBox)
        self.DeleteFileButton.setObjectName("DeleteFileButton")
        self.DeleteFileButton.setFixedWidth(75)
        self.DeleteFileButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.DeleteFileButton.setText("Delete")
        self.DeleteFileButton.clicked.connect(self.delete_file)
        
        #Neural network training and testing
        self.CNNBox = QtWidgets.QGroupBox(self)
        self.CNNBox.setObjectName("CNNBox")
        self.CNNBox.setFixedWidth(DataSelBoxWidth)
        self.CNNBox.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        self.CNNBox.setTitle("Neural network")
        
        #Train CNN
        self.Label_train_CNN = QtWidgets.QLabel(self.CNNBox)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.Label_train_CNN.setFont(font)
        self.Label_train_CNN.setObjectName("Label_train_CNN")
        self.Label_train_CNN.setText("Train convolutional neural network")
        
        #Select general directory to feed CNN
        self.Label_CNN_Feed_Dir = QtWidgets.QLabel(self.CNNBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_CNN_Feed_Dir.setFont(font)
        self.Label_CNN_Feed_Dir.setObjectName("Label_CNN_Feed_Dir")
        self.Label_CNN_Feed_Dir.setText("General feed data folder: ")
        
        self.CNN_Feed_Dir = QtWidgets.QLineEdit(self.CNNBox)
        self.CNN_Feed_Dir.setFixedHeight(25)
        self.CNN_Feed_Dir.setText(str(QtCore.QDir.currentPath()+'\speech\GENERAL\CSV'))
        
        self.CNN_Feed_Dir_CommandButton = QtWidgets.QPushButton(self.CNNBox)
        self.CNN_Feed_Dir_CommandButton.setObjectName("CNN_Feed_Dir_CommandButton")
        self.CNN_Feed_Dir_CommandButton.setFixedWidth(80)
        self.CNN_Feed_Dir_CommandButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.CNN_Feed_Dir_CommandButton.setText("Open")
        self.CNN_Feed_Dir_CommandButton.clicked.connect(self.CNN_Feed_Dir_manual_change)
        
        #Select personal directory to feed CNN
        self.Label_CNN_Personalize_Dir = QtWidgets.QLabel(self.CNNBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_CNN_Personalize_Dir.setFont(font)
        self.Label_CNN_Personalize_Dir.setObjectName("Label_CNN_Personalize_Dir")
        self.Label_CNN_Personalize_Dir.setText("Personalized feed data folder: ")
        
        self.CNN_Personalize_Dir = QtWidgets.QLineEdit(self.CNNBox)
        self.CNN_Personalize_Dir.setFixedHeight(25)
        self.CNN_Personalize_Dir.setText(str(QtCore.QDir.currentPath()+'\speech\PERSONAL\CSV'))
        
        self.CNN_Personalize_Dir_CommandButton = QtWidgets.QPushButton(self.CNNBox)
        self.CNN_Personalize_Dir_CommandButton.setObjectName("CNN_Personalize_Dir_CommandButton")
        self.CNN_Personalize_Dir_CommandButton.setFixedWidth(80)
        self.CNN_Personalize_Dir_CommandButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.CNN_Personalize_Dir_CommandButton.setText("Open")
        self.CNN_Personalize_Dir_CommandButton.clicked.connect(self.CNN_Personalize_Dir_manual_change)
        
        #Personal voice use percentage
        self.Label_Personal_Use = QtWidgets.QLabel(self.CNNBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_Personal_Use.setFont(font)
        self.Label_Personal_Use.setObjectName("Label_Personal_Use")
        self.Label_Personal_Use.setText("Use of personalized data:   ")
        
        self.Personal_Use_horizontalSlider = QtWidgets.QSlider(self.CNNBox)
        self.Personal_Use_horizontalSlider.setMinimum(0)
        self.Personal_Use_horizontalSlider.setMaximum(100)
        self.Personal_Use_horizontalSlider.setSingleStep(1)
        self.Personal_Use_horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.Personal_Use_horizontalSlider.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.Personal_Use_horizontalSlider.setTickInterval(5)
        self.Personal_Use_horizontalSlider.setObjectName("Personal_Use_horizontalSlider")
        self.Personal_Use_horizontalSlider.setValue(0)
        self.Personal_Use_horizontalSlider.valueChanged.connect(self.personal_data_feed_manual_change)
        
        self.Label_Personal_Use_Value = QtWidgets.QLabel(self.CNNBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_Personal_Use_Value.setFont(font)
        self.Label_Personal_Use_Value.setObjectName("Label_Personal_Use_Value")
        self.Label_Personal_Use_Value.setText(" " + str(personal_data_feed_slider) + " %")
        
        #Specify CNN traning batch size
        self.Label_CNN_Batches = QtWidgets.QLabel(self.CNNBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_CNN_Batches.setFont(font)
        self.Label_CNN_Batches.setObjectName("Label_CNN_Batches")
        self.Label_CNN_Batches.setText("CNN traning batch size: ")
        
        self.CNN_Batches = QtWidgets.QLineEdit(self.CNNBox)
        self.CNN_Batches.setFixedHeight(25)
        self.CNN_Batches.setText(str(CNN_batches))
        self.CNN_Batches.textChanged.connect(self.CNN_Batches_manual_change)
        
        #Specify CNN traning number of epochs
        self.Label_CNN_Epochs = QtWidgets.QLabel(self.CNNBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_CNN_Epochs.setFont(font)
        self.Label_CNN_Epochs.setObjectName("Label_CNN_Epochs")
        self.Label_CNN_Epochs.setText("CNN traning number of epochs: ")
        
        self.CNN_Epochs = QtWidgets.QLineEdit(self.CNNBox)
        self.CNN_Epochs.setFixedHeight(25)
        self.CNN_Epochs.setText(str(CNN_epochs))
        self.CNN_Epochs.textChanged.connect(self.CNN_Epochs_manual_change)
        
        #Start CNN Training
        self.CNN_Train_CommandButton = QtWidgets.QPushButton(self.CNNBox)
        self.CNN_Train_CommandButton.setObjectName("CNN_Train_CommandButton")
        self.CNN_Train_CommandButton.setFixedWidth(80)
        self.CNN_Train_CommandButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.CNN_Train_CommandButton.setText("Train CNN")
        self.CNN_Train_CommandButton.clicked.connect(self.built_CNN)
        
        self.Label_CNN_Status = QtWidgets.QLabel(self.CNNBox)
        self.Label_CNN_Status.setObjectName("Label_CNN_Status")
        CNN_Status = QtGui.QPixmap(QtCore.QDir.currentPath()+'/GUI_images/CNN_inactive.png')
        self.Label_CNN_Status.setPixmap(CNN_Status.scaled(65,60))
        
        #Load and test CNN
        self.Label_LoadCNN = QtWidgets.QLabel(self.CNNBox)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.Label_LoadCNN.setFont(font)
        self.Label_LoadCNN.setObjectName("Label_LoadCNN")
        self.Label_LoadCNN.setText("Load convolutional neural network")
        
        #Microphone selection
        self.Label_CNNMicSelection = QtWidgets.QLabel(self.CNNBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_CNNMicSelection.setFont(font)
        self.Label_CNNMicSelection.setObjectName("Label_CNNMicSelection")
        self.Label_CNNMicSelection.setText("Microphone:")
        
        self.CNNMicSelectionDropDown=QtWidgets.QComboBox(self.CNNBox)
        self.CNNMicSelectionDropDown.addItems([""])
        self.CNNMicSelectionDropDown.setMinimumWidth(80)

        #Microphone volume
        self.Label_CNNMicVolume = QtWidgets.QLabel(self.CNNBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_CNNMicVolume.setFont(font)
        self.Label_CNNMicVolume.setObjectName("Label_CNNMicVolume")
        self.Label_CNNMicVolume.setText("Microphone volume:")
        
        self.CNNmic_levelBar = QtWidgets.QProgressBar(self.CNNBox)
        self.CNNmic_levelBar.setProperty("value", 0)
        self.CNNmic_levelBar.setObjectName("CNNmic_levelBar")
        self.CNNmic_levelBar.setFixedHeight(6)
        self.CNNmic_levelBar.setTextVisible(False)
        self.CNNmic_levelBar.setSizePolicy(QtWidgets.QSizePolicy.Expanding,QtWidgets.QSizePolicy.Expanding)
        
        self.Label_CNNMicVolume_Value = QtWidgets.QLabel(self.CNNBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_CNNMicVolume_Value.setFont(font)
        self.Label_CNNMicVolume_Value.setObjectName("Label_CNNMicVolume_Value")
        self.Label_CNNMicVolume_Value.setText(" " + str(0) + " %")
        
        #Background noise threshold
        self.Label_cnnMicThreshold = QtWidgets.QLabel(self.CNNBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_cnnMicThreshold.setFont(font)
        self.Label_cnnMicThreshold.setObjectName("Label_cnnMicThreshold")
        self.Label_cnnMicThreshold.setText("Silence threshold:   ")
        
        self.cnn_mic_horizontalSlider = QtWidgets.QSlider(self.CNNBox)
        self.cnn_mic_horizontalSlider.setMinimum(0)
        self.cnn_mic_horizontalSlider.setMaximum(100)
        self.cnn_mic_horizontalSlider.setSingleStep(1)
        self.cnn_mic_horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.cnn_mic_horizontalSlider.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.cnn_mic_horizontalSlider.setTickInterval(5)
        self.cnn_mic_horizontalSlider.setObjectName("cnn_mic_horizontalSlider")
        self.cnn_mic_horizontalSlider.setValue(0)
        self.cnn_mic_horizontalSlider.valueChanged.connect(self.CNN_silence_threshold_manual_change)
        
        self.Label_cnnMicThreshold_Value = QtWidgets.QLabel(self.CNNBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_cnnMicThreshold_Value.setFont(font)
        self.Label_cnnMicThreshold_Value.setObjectName("Label_cnnMicThreshold_Value")
        self.Label_cnnMicThreshold_Value.setText(" " + str(cnn_silence_treshold_slider) + " %")
        
        self.cnnCheckSilenceCommandButton = QtWidgets.QPushButton(self.CNNBox)
        self.cnnCheckSilenceCommandButton.setObjectName("cnnCheckSilenceCommandButton")
        self.cnnCheckSilenceCommandButton.setFixedWidth(80)
        self.cnnCheckSilenceCommandButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.cnnCheckSilenceCommandButton.setText("Check")
        self.cnnCheckSilenceCommandButton.clicked.connect(self.CNN_check_silence_threshold)
        
        #Select target directory for CNN loading
        self.Label_CNN_File = QtWidgets.QLabel(self.CNNBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_CNN_File.setFont(font)
        self.Label_CNN_File.setObjectName("Label_CNN_File")
        self.Label_CNN_File.setText("Select CNN: ")
        
        self.CNN_File = QtWidgets.QLineEdit(self.CNNBox)
        self.CNN_File.setFixedHeight(25)
        
        self.CNN_File_CommandButton = QtWidgets.QPushButton(self.CNNBox)
        self.CNN_File_CommandButton.setObjectName("CNN_File_CommandButton")
        self.CNN_File_CommandButton.setFixedWidth(80)
        self.CNN_File_CommandButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.CNN_File_CommandButton.setText("Open")
        self.CNN_File_CommandButton.clicked.connect(self.CNN_File_manual_change)
        
        #CNN loading and test control
        self.Label_CNNTestStatus = QtWidgets.QLabel(self.CNNBox)
        self.Label_CNNTestStatus.setObjectName("Label_CNNTestStatus")
        CNNTestStatus = QtGui.QPixmap(QtCore.QDir.currentPath()+'/GUI_images/CNN_test_inactive.png')
        self.Label_CNNTestStatus.setPixmap(CNNTestStatus.scaled(22,22))
        
        self.CNN_Load_File_CommandButton = QtWidgets.QPushButton(self.CNNBox)
        self.CNN_Load_File_CommandButton.setObjectName("CNN_Load_File_CommandButton")
        self.CNN_Load_File_CommandButton.setFixedWidth(80)
        self.CNN_Load_File_CommandButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.CNN_Load_File_CommandButton.setText("Load CNN")
        self.CNN_Load_File_CommandButton.clicked.connect(self.CNN_Load_network)
        
        self.CNN_Test_CommandButton = QtWidgets.QPushButton(self.CNNBox)
        self.CNN_Test_CommandButton.setObjectName("CNN_Test_CommandButton")
        self.CNN_Test_CommandButton.setFixedWidth(80)
        self.CNN_Test_CommandButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.CNN_Test_CommandButton.setText("Test")
        self.CNN_Test_CommandButton.clicked.connect(self.CNN_Test_network)
        self.CNN_Test_CommandButton.setEnabled(False)
        
        #CNN display result
        self.Label_CNN_Result = QtWidgets.QLabel(self.CNNBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_CNN_Result.setFont(font)
        self.Label_CNN_Result.setObjectName("Label_CNN_Result")
        self.Label_CNN_Result.setText("Recognized word: ")
        
        self.CNN_Result = QtWidgets.QLineEdit(self.CNNBox)
        self.CNN_Result.setFixedHeight(25)
        
        #STFT preview
        self.DataRepBox = QtWidgets.QGroupBox(self)
        self.DataRepBox.setObjectName("DataRepBox")
        self.DataRepBox.setTitle("Short Time Fourier Transform Plot")
        
        self.Plt=plt.figure()
        self.canvas=FigureCanvasQTAgg(self.Plt)
        
        #Control box
        self.SpeechControlBox = QtWidgets.QGroupBox(self)
        self.SpeechControlBox.setObjectName("SpeechControlBox")
        
        self.SpeechControlBox.setFixedWidth(SpeechControlBoxWidth)
        self.SpeechControlBox.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        self.SpeechControlBox.setTitle("Data augmentation")
        
        #Record voice command
        self.Label_VoiceCommand = QtWidgets.QLabel(self.SpeechControlBox)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.Label_VoiceCommand.setFont(font)
        self.Label_VoiceCommand.setObjectName("Label_VoiceCommand")
        self.Label_VoiceCommand.setText("Record voice command")
        
        #Microphone selection
        self.Label_MicSelection = QtWidgets.QLabel(self.SpeechControlBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_MicSelection.setFont(font)
        self.Label_MicSelection.setObjectName("Label_MicSelection")
        self.Label_MicSelection.setText("Microphone:")
        
        self.MicSelectionDropDown=QtWidgets.QComboBox(self.SpeechControlBox)
        self.MicSelectionDropDown.addItems([""])
        self.MicSelectionDropDown.setMinimumWidth(80)

        #Microphone volume
        self.Label_MicVolume = QtWidgets.QLabel(self.SpeechControlBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_MicVolume.setFont(font)
        self.Label_MicVolume.setObjectName("Label_MicVolume")
        self.Label_MicVolume.setText("Microphone volume:")
        
        self.mic_levelBar = QtWidgets.QProgressBar(self)
        self.mic_levelBar.setProperty("value", 0)
        self.mic_levelBar.setObjectName("mic_levelBar")
        self.mic_levelBar.setFixedHeight(6)
        self.mic_levelBar.setTextVisible(False)
        self.mic_levelBar.setSizePolicy(QtWidgets.QSizePolicy.Expanding,QtWidgets.QSizePolicy.Expanding)
        
        self.Label_MicVolume_Value = QtWidgets.QLabel(self.SpeechControlBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_MicVolume_Value.setFont(font)
        self.Label_MicVolume_Value.setObjectName("Label_MicVolume_Value")
        self.Label_MicVolume_Value.setText(" " + str(0) + " %")
        
        #Recording format selection
        self.Label_RecordToFormat = QtWidgets.QLabel(self.SpeechControlBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_RecordToFormat.setFont(font)
        self.Label_RecordToFormat.setObjectName("Label_RecordToFormat")
        self.Label_RecordToFormat.setText("Save in format:")
        
        self.FormatSelBox = QtWidgets.QGroupBox(self.SpeechControlBox)
        self.FormatSelBox.setObjectName("FormatSelBox")
        self.FormatSelBox.setFixedWidth(150)
        self.FormatSelBox.setFixedHeight(35)
        
        self.Label_RecordToFormatWAV = QtWidgets.QRadioButton(self.FormatSelBox)
        self.Label_RecordToFormatWAV.setObjectName("Label_RecordToFormatWAV")
        self.Label_RecordToFormatWAV.setText("WAV")
        if audio_recording_output_format == 0:
            self.Label_RecordToFormatWAV.setChecked(True)
        else:
            self.Label_RecordToFormatWAV.setChecked(False)
        self.Label_RecordToFormatWAV.clicked.connect(self.audio_recording_output_format_manual_change)
        
        self.Label_RecordToFormatCSV = QtWidgets.QRadioButton(self.FormatSelBox)
        self.Label_RecordToFormatCSV.setObjectName("Label_RecordToFormatCSV")
        self.Label_RecordToFormatCSV.setText("CSV")
        if audio_recording_output_format == 0:
            self.Label_RecordToFormatCSV.setChecked(False)
        else:
            self.Label_RecordToFormatCSV.setChecked(True)
        self.Label_RecordToFormatCSV.clicked.connect(self.audio_recording_output_format_manual_change)
        
        #Voice command selection
        self.Label_CommandList = QtWidgets.QLabel(self.SpeechControlBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_CommandList.setFont(font)
        self.Label_CommandList.setObjectName("Label_CommandList")
        self.Label_CommandList.setText("Select command:")
        
        self.CommandList=QtWidgets.QComboBox(self.SpeechControlBox)
        self.CommandList.addItems([""])
        self.CommandList.setMinimumWidth(80)
        
        #Number of recordings
        self.Label_Repeats = QtWidgets.QLabel(self.SpeechControlBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_Repeats.setFont(font)
        self.Label_Repeats.setObjectName("Label_Repeats")
        self.Label_Repeats.setText("Number of command repeats:")
        
        self.Repeats=QtWidgets.QComboBox(self.SpeechControlBox)
        self.Repeats.addItems(["1", "2", "3", "5", "10", "15", "20"])
        self.Repeats.setMinimumWidth(80)
        self.Repeats.setCurrentText(str(command_repeats))
        self.Repeats.currentTextChanged.connect(self.repeats_manual_change)
        
        #Command duration
        self.Label_CommandDuration = QtWidgets.QLabel(self.SpeechControlBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_CommandDuration.setFont(font)
        self.Label_CommandDuration.setObjectName("Label_CommandDuration")
        self.Label_CommandDuration.setText("Duration, ms")
        
        self.command_duration_horizontalSlider = QtWidgets.QSlider(self.SpeechControlBox)
        self.command_duration_horizontalSlider.setMinimum(100)
        self.command_duration_horizontalSlider.setMaximum(1000)
        self.command_duration_horizontalSlider.setSingleStep(100)
        self.command_duration_horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.command_duration_horizontalSlider.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.command_duration_horizontalSlider.setTickInterval(100)
        self.command_duration_horizontalSlider.setObjectName("command_duration_horizontalSlider")
        self.command_duration_horizontalSlider.setValue(command_duration*1000)
        self.command_duration_horizontalSlider.valueChanged.connect(self.command_duration_manual_change)
        
        self.Label_CommandDuration_Value = QtWidgets.QLabel(self.SpeechControlBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_CommandDuration_Value.setFont(font)
        self.Label_CommandDuration_Value.setObjectName("Label_CommandDuration_Value")
        self.Label_CommandDuration_Value.setText(" " + str(int(command_duration*1000)) + " ms")
        
        #Audio rate
        self.Label_RATE = QtWidgets.QLabel(self.SpeechControlBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_RATE.setFont(font)
        self.Label_RATE.setObjectName("Label_RATE")
        self.Label_RATE.setText("Audio rate, Hz:")
        
        self.RATE_Value=QtWidgets.QComboBox(self.SpeechControlBox)
        self.RATE_Value.addItems(["22050", "32000", "44100", "48000"])
        self.RATE_Value.setMinimumWidth(80)
        self.RATE_Value.setCurrentText(str(RATE))
        self.RATE_Value.currentTextChanged.connect(self.RATE_manual_change)
        
        #STFT segment size
        self.Label_SegmentSize = QtWidgets.QLabel(self.SpeechControlBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_SegmentSize.setFont(font)
        self.Label_SegmentSize.setObjectName("Label_SegmentSize")
        self.Label_SegmentSize.setText("STFT Segment size:")
        
        self.stft_segment_size_horizontalSlider = QtWidgets.QSlider(self.SpeechControlBox)
        self.stft_segment_size_horizontalSlider.setMinimum(100)
        self.stft_segment_size_horizontalSlider.setMaximum(3000)
        self.stft_segment_size_horizontalSlider.setSingleStep(100)
        self.stft_segment_size_horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.stft_segment_size_horizontalSlider.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.stft_segment_size_horizontalSlider.setTickInterval(100)
        self.stft_segment_size_horizontalSlider.setObjectName("stft_segment_size_horizontalSlider")
        self.stft_segment_size_horizontalSlider.setValue(stft_segment_size)
        self.stft_segment_size_horizontalSlider.valueChanged.connect(self.stft_segment_size_manual_change)
        
        self.Label_SegmentSize_Value = QtWidgets.QLabel(self.SpeechControlBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_SegmentSize_Value.setFont(font)
        self.Label_SegmentSize_Value.setObjectName("Label_SegmentSize_Value")
        self.Label_SegmentSize_Value.setText(" " + str(stft_segment_size))
        
        #STFT frequency resolution
        self.Label_stft_freq_res = QtWidgets.QLabel(self.SpeechControlBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_stft_freq_res.setFont(font)
        self.Label_stft_freq_res.setObjectName("Label_stft_freq_res")
        self.Label_stft_freq_res.setText("STFT Frequency resolution:")
        
        self.stft_freq_res_horizontalSlider = QtWidgets.QSlider(self.SpeechControlBox)
        self.stft_freq_res_horizontalSlider.setMinimum(5)
        self.stft_freq_res_horizontalSlider.setMaximum(100)
        self.stft_freq_res_horizontalSlider.setSingleStep(5)
        self.stft_freq_res_horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.stft_freq_res_horizontalSlider.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.stft_freq_res_horizontalSlider.setTickInterval(10)
        self.stft_freq_res_horizontalSlider.setObjectName("stft_freq_res_horizontalSlider")
        self.stft_freq_res_horizontalSlider.setValue(stft_freq_res)
        self.stft_freq_res_horizontalSlider.valueChanged.connect(self.stft_freq_res_manual_change)
        
        self.Label_stft_freq_res_Value = QtWidgets.QLabel(self.SpeechControlBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_stft_freq_res_Value.setFont(font)
        self.Label_stft_freq_res_Value.setObjectName("Label_stft_freq_res_Value")
        self.Label_stft_freq_res_Value.setText(" " + str(stft_freq_res) + " Hz")
        
        #STFT overlap percent
        self.Label_stft_overlap_percent = QtWidgets.QLabel(self.SpeechControlBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_stft_overlap_percent.setFont(font)
        self.Label_stft_overlap_percent.setObjectName("Label_stft_overlap_percent")
        self.Label_stft_overlap_percent.setText("STFT Segments overlap:")
        
        self.stft_overlap_percent_horizontalSlider = QtWidgets.QSlider(self.SpeechControlBox)
        self.stft_overlap_percent_horizontalSlider.setMinimum(0)
        self.stft_overlap_percent_horizontalSlider.setMaximum(99)
        self.stft_overlap_percent_horizontalSlider.setSingleStep(5)
        self.stft_overlap_percent_horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.stft_overlap_percent_horizontalSlider.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.stft_overlap_percent_horizontalSlider.setTickInterval(10)
        self.stft_overlap_percent_horizontalSlider.setObjectName("stft_overlap_percent_horizontalSlider")
        self.stft_overlap_percent_horizontalSlider.setValue(stft_overlap_percent)
        self.stft_overlap_percent_horizontalSlider.valueChanged.connect(self.stft_overlap_percent_manual_change)
        
        self.Label_stft_overlap_percent_Value = QtWidgets.QLabel(self.SpeechControlBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_stft_overlap_percent_Value.setFont(font)
        self.Label_stft_overlap_percent_Value.setObjectName("Label_stft_overlap_percent_Value")
        self.Label_stft_overlap_percent_Value.setText(" " + str(stft_overlap_percent) + " %")

        #STFT frequency high limit
        self.Label_stft_freq_high_lim = QtWidgets.QLabel(self.SpeechControlBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_stft_freq_high_lim.setFont(font)
        self.Label_stft_freq_high_lim.setObjectName("Label_stft_freq_high_lim")
        self.Label_stft_freq_high_lim.setText("STFT Frequency high limit:")
        
        self.stft_freq_high_lim_horizontalSlider = QtWidgets.QSlider(self.SpeechControlBox)
        self.stft_freq_high_lim_horizontalSlider.setMinimum(2000)
        self.stft_freq_high_lim_horizontalSlider.setMaximum(24000)
        self.stft_freq_high_lim_horizontalSlider.setSingleStep(500)
        self.stft_freq_high_lim_horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.stft_freq_high_lim_horizontalSlider.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.stft_freq_high_lim_horizontalSlider.setTickInterval(2000)
        self.stft_freq_high_lim_horizontalSlider.setObjectName("stft_freq_high_lim_horizontalSlider")
        self.stft_freq_high_lim_horizontalSlider.setValue(stft_freq_high_lim)
        self.stft_freq_high_lim_horizontalSlider.valueChanged.connect(self.stft_freq_high_lim_manual_change)
        
        self.Label_stft_freq_high_lim_Value = QtWidgets.QLabel(self.SpeechControlBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_stft_freq_high_lim_Value.setFont(font)
        self.Label_stft_freq_high_lim_Value.setObjectName("Label_stft_freq_high_lim_Value")
        self.Label_stft_freq_high_lim_Value.setText(" " + str(stft_freq_high_lim) + " Hz")
        
        #STFT data normalization
        self.Label_stft_data_norm = QtWidgets.QLabel(self.SpeechControlBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_stft_data_norm.setFont(font)
        self.Label_stft_data_norm.setObjectName("Label_stft_data_norm")
        self.Label_stft_data_norm.setText("STFT data normalization:")
        
        self.NormSelBox = QtWidgets.QGroupBox(self)
        self.NormSelBox.setObjectName("NormSelBox")
        self.NormSelBox.setFixedWidth(150)
        self.NormSelBox.setFixedHeight(35)
        
        self.Label_stft_data_norm_LIN = QtWidgets.QRadioButton(self.NormSelBox)
        self.Label_stft_data_norm_LIN.setObjectName("Label_stft_data_norm_LIN")
        self.Label_stft_data_norm_LIN.setText("LIN")
        if stft_lin_log_norm == 0:
            self.Label_stft_data_norm_LIN.setChecked(True)
        else:
            self.Label_stft_data_norm_LIN.setChecked(False)
        self.Label_stft_data_norm_LIN.clicked.connect(self.stft_data_norm_manual_change)
        
        self.Label_stft_data_norm_LOG = QtWidgets.QRadioButton(self.NormSelBox)
        self.Label_stft_data_norm_LOG.setObjectName("Label_stft_data_norm_LOG")
        self.Label_stft_data_norm_LOG.setText("LOG")
        if stft_lin_log_norm == 0:
            self.Label_stft_data_norm_LOG.setChecked(False)
        else:
            self.Label_stft_data_norm_LOG.setChecked(True)
        self.Label_stft_data_norm_LOG.setChecked(False)
        self.Label_stft_data_norm_LOG.clicked.connect(self.stft_data_norm_manual_change)
        
        #Data batches frequency
        self.Label_FREQ = QtWidgets.QLabel(self.SpeechControlBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_FREQ.setFont(font)
        self.Label_FREQ.setObjectName("Label_FREQ")
        self.Label_FREQ.setText("Data batches frequency:  ")
        
        self.Label_FREQ_Value = QtWidgets.QLabel(self.SpeechControlBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_FREQ_Value.setFont(font)
        self.Label_FREQ_Value.setObjectName("Label_FREQ_Value")
        self.Label_FREQ_Value.setText(str(int(FREQ)) + " Hz")
        
        #Size of audio data chunks
        self.Label_CHUNK = QtWidgets.QLabel(self.SpeechControlBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_CHUNK.setFont(font)
        self.Label_CHUNK.setObjectName("Label_CHUNK")
        self.Label_CHUNK.setText("Size of audio data chunks:  ")
        
        self.Label_CHUNK_Value = QtWidgets.QLabel(self.SpeechControlBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_CHUNK_Value.setFont(font)
        self.Label_CHUNK_Value.setObjectName("Label_CHUNK_Value")
        self.Label_CHUNK_Value.setText(str(int(CHUNK)))
        
        #Size of command data chunks
        self.Label_COMMAND_CHUNK = QtWidgets.QLabel(self.SpeechControlBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_COMMAND_CHUNK.setFont(font)
        self.Label_COMMAND_CHUNK.setObjectName("Label_COMMAND_CHUNK")
        self.Label_COMMAND_CHUNK.setText("Size of command data chunks:  ")
        
        self.Label_COMMAND_CHUNK_Value = QtWidgets.QLabel(self.SpeechControlBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_COMMAND_CHUNK_Value.setFont(font)
        self.Label_COMMAND_CHUNK_Value.setObjectName("Label_COMMAND_CHUNK_Value")
        self.Label_COMMAND_CHUNK_Value.setText(str(int(COMMAND_CHUNK)))
        
        #Background noise threshold
        self.Label_MicThreshold = QtWidgets.QLabel(self.SpeechControlBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_MicThreshold.setFont(font)
        self.Label_MicThreshold.setObjectName("Label_MicThreshold")
        self.Label_MicThreshold.setText("Silence threshold:   ")
        
        self.mic_horizontalSlider = QtWidgets.QSlider(self.SpeechControlBox)
        self.mic_horizontalSlider.setMinimum(0)
        self.mic_horizontalSlider.setMaximum(100)
        self.mic_horizontalSlider.setSingleStep(1)
        self.mic_horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.mic_horizontalSlider.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.mic_horizontalSlider.setTickInterval(5)
        self.mic_horizontalSlider.setObjectName("mic_horizontalSlider")
        self.mic_horizontalSlider.setValue(0)
        self.mic_horizontalSlider.valueChanged.connect(self.silence_threshold_manual_change)
        
        self.Label_MicThreshold_Value = QtWidgets.QLabel(self.SpeechControlBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_MicThreshold_Value.setFont(font)
        self.Label_MicThreshold_Value.setObjectName("Label_MicThreshold_Value")
        self.Label_MicThreshold_Value.setText(" " + str(silence_treshold_slider) + " %")
        
        self.CheckSilenceCommandButton = QtWidgets.QPushButton(self.SpeechControlBox)
        self.CheckSilenceCommandButton.setObjectName("CheckSilenceCommandButton")
        self.CheckSilenceCommandButton.setFixedWidth(80)
        self.CheckSilenceCommandButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.CheckSilenceCommandButton.setText("Check")
        self.CheckSilenceCommandButton.clicked.connect(self.check_silence_threshold)
        
        #Recording status and control
        self.Label_VoiceRecordStatus = QtWidgets.QLabel(self.SpeechControlBox)
        self.Label_VoiceRecordStatus.setObjectName("Label_VoiceRecordStatus")
        VoiceRecordStatus = QtGui.QPixmap(QtCore.QDir.currentPath()+'/GUI_images/recording_inactive.png')
        self.Label_VoiceRecordStatus.setPixmap(VoiceRecordStatus.scaled(22,22))
        
        self.StartRecordCommandButton = QtWidgets.QPushButton(self.SpeechControlBox)
        self.StartRecordCommandButton.setObjectName("StartRecordCommandButton")
        self.StartRecordCommandButton.setFixedWidth(120)
        self.StartRecordCommandButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.StartRecordCommandButton.setText("Record")
        self.StartRecordCommandButton.clicked.connect(self.run_command_recording)
        
        #Convert wav to csv
        self.Label_WAV_to_CSV = QtWidgets.QLabel(self.SpeechControlBox)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.Label_WAV_to_CSV.setFont(font)
        self.Label_WAV_to_CSV.setObjectName("Label_WAV_to_CSV")
        self.Label_WAV_to_CSV.setText("Process WAV to STFT spectrogram")
        
        #Select source directory for WAV to CSV conversion
        self.Label_WAV_to_CSV_Source_Dir = QtWidgets.QLabel(self.SpeechControlBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_WAV_to_CSV_Source_Dir.setFont(font)
        self.Label_WAV_to_CSV_Source_Dir.setObjectName("Label_WAV_to_CSV_Source_Dir")
        self.Label_WAV_to_CSV_Source_Dir.setText("Source directory: ")
        
        self.WAV_to_CSV_Source_Dir = QtWidgets.QLineEdit(self.SpeechControlBox)
        self.WAV_to_CSV_Source_Dir.setFixedHeight(25)
        
        self.WAV_to_CSV_Source_Dir_CommandButton = QtWidgets.QPushButton(self.SpeechControlBox)
        self.WAV_to_CSV_Source_Dir_CommandButton.setObjectName("WAV_to_CSV_Source_Dir_CommandButton")
        self.WAV_to_CSV_Source_Dir_CommandButton.setFixedWidth(80)
        self.WAV_to_CSV_Source_Dir_CommandButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.WAV_to_CSV_Source_Dir_CommandButton.setText("Open")
        self.WAV_to_CSV_Source_Dir_CommandButton.clicked.connect(self.WAV_to_CSV_Source_Dir_manual_change)
        
        #Select target directory for WAV to CSV conversion
        self.Label_WAV_to_CSV_Target_Dir = QtWidgets.QLabel(self.SpeechControlBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_WAV_to_CSV_Target_Dir.setFont(font)
        self.Label_WAV_to_CSV_Target_Dir.setObjectName("Label_WAV_to_CSV_Target_Dir")
        self.Label_WAV_to_CSV_Target_Dir.setText("Target directory: ")
        
        self.WAV_to_CSV_Target_Dir = QtWidgets.QLineEdit(self.SpeechControlBox)
        self.WAV_to_CSV_Target_Dir.setFixedHeight(25)
        
        self.WAV_to_CSV_Target_Dir_CommandButton = QtWidgets.QPushButton(self.SpeechControlBox)
        self.WAV_to_CSV_Target_Dir_CommandButton.setObjectName("WAV_to_CSV_Target_Dir_CommandButton")
        self.WAV_to_CSV_Target_Dir_CommandButton.setFixedWidth(80)
        self.WAV_to_CSV_Target_Dir_CommandButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.WAV_to_CSV_Target_Dir_CommandButton.setText("Open")
        self.WAV_to_CSV_Target_Dir_CommandButton.clicked.connect(self.WAV_to_CSV_Target_Dir_manual_change)
        
        #Start conversion
        self.WAV_to_CSV_Convert_CommandButton = QtWidgets.QPushButton(self.SpeechControlBox)
        self.WAV_to_CSV_Convert_CommandButton.setObjectName("WAV_to_CSV_Convert_CommandButton")
        self.WAV_to_CSV_Convert_CommandButton.setFixedWidth(80)
        self.WAV_to_CSV_Convert_CommandButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.WAV_to_CSV_Convert_CommandButton.setText("Convert")
        self.WAV_to_CSV_Convert_CommandButton.clicked.connect(self.converting_wav_to_csv)
        
        self.Label_WAV_to_CSV_Convert_Status = QtWidgets.QLabel(self.SpeechControlBox)
        self.Label_WAV_to_CSV_Convert_Status.setObjectName("Label_WAV_to_CSV_Convert_Status")
        WAV_to_CSV_Convert_Status = QtGui.QPixmap(QtCore.QDir.currentPath()+'/GUI_images/wav_to_csv_inactive.png')
        self.Label_WAV_to_CSV_Convert_Status.setPixmap(WAV_to_CSV_Convert_Status.scaled(65,60))
        
        #Filter spectrogram
        self.Label_filter_spectrogram = QtWidgets.QLabel(self.SpeechControlBox)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.Label_filter_spectrogram.setFont(font)
        self.Label_filter_spectrogram.setObjectName("Label_filter_spectrogram")
        self.Label_filter_spectrogram.setText("Filter spectrograms")
        
        #Select directory for filtering
        self.Label_Filt_Spectr_Dir = QtWidgets.QLabel(self.SpeechControlBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_Filt_Spectr_Dir.setFont(font)
        self.Label_Filt_Spectr_Dir.setObjectName("Label_Filt_Spectr_Dir")
        self.Label_Filt_Spectr_Dir.setText("Target folder: ")
        
        self.Filt_Spectr_Dir = QtWidgets.QLineEdit(self.SpeechControlBox)
        self.Filt_Spectr_Dir.setFixedHeight(25)
        self.Filt_Spectr_Dir.setText(str(QtCore.QDir.currentPath()+'\speech\GENERAL\CSV'))
        
        self.Filt_Spectr_Dir_CommandButton = QtWidgets.QPushButton(self.SpeechControlBox)
        self.Filt_Spectr_Dir_CommandButton.setObjectName("Filt_Spectr_Dir_CommandButton")
        self.Filt_Spectr_Dir_CommandButton.setFixedWidth(80)
        self.Filt_Spectr_Dir_CommandButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.Filt_Spectr_Dir_CommandButton.setText("Open")
        self.Filt_Spectr_Dir_CommandButton.clicked.connect(self.Filt_Spectr_Dir_manual_change)
        
        #Run filtering
        self.Filt_Spectr_CommandButton = QtWidgets.QPushButton(self.SpeechControlBox)
        self.Filt_Spectr_CommandButton.setObjectName("Filt_Spectr_CommandButton")
        self.Filt_Spectr_CommandButton.setFixedWidth(80)
        self.Filt_Spectr_CommandButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.Filt_Spectr_CommandButton.setText("Filter")
        self.Filt_Spectr_CommandButton.clicked.connect(self.filter_spectrogram)
        
        self.Label_Filt_Spectr_Status = QtWidgets.QLabel(self.SpeechControlBox)
        self.Label_Filt_Spectr_Status.setObjectName("Label_Filt_Spectr_Status")
        Filt_Spectr_Status = QtGui.QPixmap(QtCore.QDir.currentPath()+'/GUI_images/filter_inactive.png')
        self.Label_Filt_Spectr_Status.setPixmap(Filt_Spectr_Status.scaled(42,40))
        
        #Progress bar
        self.progressBar = QtWidgets.QProgressBar(self)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.progressBar.setFixedHeight(10)
        self.progressBar.setSizePolicy(QtWidgets.QSizePolicy.Expanding,QtWidgets.QSizePolicy.Expanding)
        
        self.populate_mic_selection()
        self.audio_recording_output_format_manual_change()
        self.command_duration_programmable_change()
        
    def __layout(self):
        self.vmainbox=QtWidgets.QVBoxLayout()       
        self.hmenubox = QtWidgets.QHBoxLayout()
        self.hmenubox.addWidget(self.menu_bar)
        self.hgroupbox = QtWidgets.QHBoxLayout()
        self.vgroupbox = QtWidgets.QVBoxLayout()
        self.vgroupbox.addWidget(self.DataSelBox)
        self.vgroupbox.addWidget(self.CNNBox)
        self.hgroupbox.addLayout(self.vgroupbox)
        self.hgroupbox.addWidget(self.DataRepBox)
        self.hgroupbox.addWidget(self.SpeechControlBox)
        self.vmainbox.addLayout(self.hmenubox)
        self.vmainbox.addLayout(self.hgroupbox)
        self.vmainbox.addWidget(self.progressBar)
        self.setLayout(self.vmainbox)
        
        #File tree
        self.vDataSelBox=QtWidgets.QVBoxLayout()
        self.h1DataSelBox=QtWidgets.QHBoxLayout()
        
        self.h1DataSelBox.addWidget(self.DeleteFileButton)
        self.h1DataSelBox.addWidget(self.AddPlotButton)
        
        self.vDataSelBox.addWidget(self.treeView)
        self.vDataSelBox.addLayout(self.h1DataSelBox)
        
        self.vDataSelBox.setAlignment(self.h1DataSelBox, QtCore.Qt.AlignCenter)
        self.DataSelBox.setLayout(self.vDataSelBox)
        
        #CNN training and testing
        self.vCNNBox=QtWidgets.QVBoxLayout()
        
        self.h1CNNBox=QtWidgets.QHBoxLayout()
        self.h2CNNBox=QtWidgets.QHBoxLayout()
        self.h3CNNBox=QtWidgets.QHBoxLayout()
        self.h4CNNBox=QtWidgets.QHBoxLayout()
        self.h5CNNBox=QtWidgets.QHBoxLayout()
        self.h6CNNBox=QtWidgets.QHBoxLayout()
        self.h7CNNBox=QtWidgets.QHBoxLayout()
        self.h8CNNBox=QtWidgets.QHBoxLayout()
        self.h9CNNBox=QtWidgets.QHBoxLayout()
        self.h10CNNBox=QtWidgets.QHBoxLayout()
        self.h11CNNBox=QtWidgets.QHBoxLayout()
        self.h12CNNBox=QtWidgets.QHBoxLayout()
        self.h13CNNBox=QtWidgets.QHBoxLayout()
        self.h14CNNBox=QtWidgets.QHBoxLayout()
        
        self.h1CNNBox.addWidget(self.Label_train_CNN)
        self.h2CNNBox.addWidget(self.Label_CNN_Feed_Dir)
        self.h2CNNBox.addWidget(self.CNN_Feed_Dir)
        self.h2CNNBox.addWidget(self.CNN_Feed_Dir_CommandButton)
        self.h3CNNBox.addWidget(self.Label_CNN_Personalize_Dir)
        self.h3CNNBox.addWidget(self.CNN_Personalize_Dir)
        self.h3CNNBox.addWidget(self.CNN_Personalize_Dir_CommandButton)
        self.h4CNNBox.addWidget(self.Label_Personal_Use)
        self.h4CNNBox.addWidget(self.Personal_Use_horizontalSlider)
        self.h4CNNBox.addWidget(self.Label_Personal_Use_Value)
        self.h5CNNBox.addWidget(self.Label_CNN_Batches)
        self.h5CNNBox.addWidget(self.CNN_Batches)
        self.h6CNNBox.addWidget(self.Label_CNN_Epochs)
        self.h6CNNBox.addWidget(self.CNN_Epochs)
        self.h7CNNBox.addWidget(self.Label_CNN_Status)
        self.h7CNNBox.addWidget(self.CNN_Train_CommandButton)
        self.h8CNNBox.addWidget(self.Label_LoadCNN)
        self.h9CNNBox.addWidget(self.Label_CNNMicSelection)
        self.h9CNNBox.addWidget(self.CNNMicSelectionDropDown)
        self.h10CNNBox.addWidget(self.Label_CNNMicVolume)
        self.h10CNNBox.addWidget(self.CNNmic_levelBar)
        self.h10CNNBox.addWidget(self.Label_CNNMicVolume_Value)
        self.h11CNNBox.addWidget(self.Label_cnnMicThreshold)
        self.h11CNNBox.addWidget(self.cnn_mic_horizontalSlider)
        self.h11CNNBox.addWidget(self.Label_cnnMicThreshold_Value)
        self.h11CNNBox.addWidget(self.cnnCheckSilenceCommandButton)
        self.h12CNNBox.addWidget(self.Label_CNN_File)
        self.h12CNNBox.addWidget(self.CNN_File)
        self.h12CNNBox.addWidget(self.CNN_File_CommandButton)
        self.h13CNNBox.addWidget(self.Label_CNNTestStatus)
        self.h13CNNBox.addWidget(self.CNN_Load_File_CommandButton)
        self.h13CNNBox.addWidget(self.CNN_Test_CommandButton)
        self.h14CNNBox.addWidget(self.Label_CNN_Result)
        self.h14CNNBox.addWidget(self.CNN_Result)
        
        self.vCNNBox.addLayout(self.h1CNNBox)
        self.vCNNBox.addLayout(self.h2CNNBox)
        self.vCNNBox.addLayout(self.h3CNNBox)
        self.vCNNBox.addLayout(self.h4CNNBox)
        self.vCNNBox.addLayout(self.h5CNNBox)
        self.vCNNBox.addLayout(self.h6CNNBox)
        self.vCNNBox.addLayout(self.h7CNNBox)
        self.vCNNBox.addLayout(self.h8CNNBox)
        self.vCNNBox.addLayout(self.h9CNNBox)
        self.vCNNBox.addLayout(self.h10CNNBox)
        self.vCNNBox.addLayout(self.h11CNNBox)
        self.vCNNBox.addLayout(self.h12CNNBox)
        self.vCNNBox.addLayout(self.h13CNNBox)
        self.vCNNBox.addLayout(self.h14CNNBox)
        self.vCNNBox.addStretch()
        
        self.CNNBox.setLayout(self.vCNNBox)
        
        #STFT preview
        self.vDataRepBox=QtWidgets.QVBoxLayout()
        self.vDataRepBox.addWidget(self.canvas)
        self.DataRepBox.setLayout(self.vDataRepBox)
        
        #Control box
        self.vSpeechControlBox=QtWidgets.QVBoxLayout()
        
        self.h1SpeechControlBox=QtWidgets.QHBoxLayout()
        self.h2SpeechControlBox=QtWidgets.QHBoxLayout()
        self.h3SpeechControlBox=QtWidgets.QHBoxLayout()
        self.h4SpeechControlBox=QtWidgets.QHBoxLayout()
        self.h5SpeechControlBox=QtWidgets.QHBoxLayout()
        self.h6SpeechControlBox=QtWidgets.QHBoxLayout()
        self.h7SpeechControlBox=QtWidgets.QHBoxLayout()
        self.h8SpeechControlBox=QtWidgets.QHBoxLayout()
        self.h9SpeechControlBox=QtWidgets.QHBoxLayout()
        self.h10SpeechControlBox=QtWidgets.QHBoxLayout()
        self.h11SpeechControlBox=QtWidgets.QHBoxLayout()
        self.h12SpeechControlBox=QtWidgets.QHBoxLayout()
        self.h13SpeechControlBox=QtWidgets.QHBoxLayout()
        self.h14SpeechControlBox=QtWidgets.QHBoxLayout()
        self.h15SpeechControlBox=QtWidgets.QHBoxLayout()
        self.h16SpeechControlBox=QtWidgets.QHBoxLayout()
        self.h17SpeechControlBox=QtWidgets.QHBoxLayout()
        self.h18SpeechControlBox=QtWidgets.QHBoxLayout()
        self.h19SpeechControlBox=QtWidgets.QHBoxLayout()
        self.h20SpeechControlBox=QtWidgets.QHBoxLayout()
        self.h21SpeechControlBox=QtWidgets.QHBoxLayout()
        self.h22SpeechControlBox=QtWidgets.QHBoxLayout()
        self.h23SpeechControlBox=QtWidgets.QHBoxLayout()
        self.h24SpeechControlBox=QtWidgets.QHBoxLayout()
        self.h25SpeechControlBox=QtWidgets.QHBoxLayout()
        
        self.v1SpeechControlBox=QtWidgets.QVBoxLayout()
        
        self.h1SpeechControlBox.addWidget(self.Label_MicSelection)
        self.h1SpeechControlBox.addWidget(self.MicSelectionDropDown)
        self.h2SpeechControlBox.addWidget(self.Label_MicVolume)
        self.h2SpeechControlBox.addWidget(self.mic_levelBar)
        self.h2SpeechControlBox.addWidget(self.Label_MicVolume_Value)
        self.hFormatSelBox = QtWidgets.QHBoxLayout()
        self.hFormatSelBox.addWidget(self.Label_RecordToFormatWAV)
        self.hFormatSelBox.addWidget(self.Label_RecordToFormatCSV)
        self.FormatSelBox.setLayout(self.hFormatSelBox)
        self.h3SpeechControlBox.addWidget(self.Label_RecordToFormat)
        self.h3SpeechControlBox.addWidget(self.FormatSelBox)
        self.h4SpeechControlBox.addWidget(self.Label_CommandList)
        self.h4SpeechControlBox.addWidget(self.CommandList)
        self.h5SpeechControlBox.addWidget(self.Label_Repeats)
        self.h5SpeechControlBox.addWidget(self.Repeats)
        self.h6SpeechControlBox.addWidget(self.Label_CommandDuration)
        self.h6SpeechControlBox.addWidget(self.command_duration_horizontalSlider)
        self.h6SpeechControlBox.addWidget(self.Label_CommandDuration_Value)
        self.h7SpeechControlBox.addWidget(self.Label_RATE)
        self.h7SpeechControlBox.addWidget(self.RATE_Value)
        self.h8SpeechControlBox.addWidget(self.Label_SegmentSize)
        self.h8SpeechControlBox.addWidget(self.stft_segment_size_horizontalSlider)
        self.h8SpeechControlBox.addWidget(self.Label_SegmentSize_Value)
        self.h9SpeechControlBox.addWidget(self.Label_stft_freq_res)
        self.h9SpeechControlBox.addWidget(self.stft_freq_res_horizontalSlider)
        self.h9SpeechControlBox.addWidget(self.Label_stft_freq_res_Value)
        self.h10SpeechControlBox.addWidget(self.Label_stft_overlap_percent)
        self.h10SpeechControlBox.addWidget(self.stft_overlap_percent_horizontalSlider)
        self.h10SpeechControlBox.addWidget(self.Label_stft_overlap_percent_Value)
        self.h11SpeechControlBox.addWidget(self.Label_stft_freq_high_lim)
        self.h11SpeechControlBox.addWidget(self.stft_freq_high_lim_horizontalSlider)
        self.h11SpeechControlBox.addWidget(self.Label_stft_freq_high_lim_Value)
        self.hNormSelBox = QtWidgets.QHBoxLayout()
        self.hNormSelBox.addWidget(self.Label_stft_data_norm_LIN)
        self.hNormSelBox.addWidget(self.Label_stft_data_norm_LOG)
        self.NormSelBox.setLayout(self.hNormSelBox)
        self.h12SpeechControlBox.addWidget(self.Label_stft_data_norm)
        self.h12SpeechControlBox.addWidget(self.NormSelBox)
        self.h13SpeechControlBox.addWidget(self.Label_FREQ)
        self.h13SpeechControlBox.addWidget(self.Label_FREQ_Value)
        self.h14SpeechControlBox.addWidget(self.Label_CHUNK)
        self.h14SpeechControlBox.addWidget(self.Label_CHUNK_Value)
        self.h15SpeechControlBox.addWidget(self.Label_COMMAND_CHUNK)
        self.h15SpeechControlBox.addWidget(self.Label_COMMAND_CHUNK_Value)
        self.h16SpeechControlBox.addWidget(self.Label_MicThreshold)
        self.h16SpeechControlBox.addWidget(self.mic_horizontalSlider)
        self.h16SpeechControlBox.addWidget(self.Label_MicThreshold_Value)
        self.h16SpeechControlBox.addWidget(self.CheckSilenceCommandButton)
        self.h17SpeechControlBox.addWidget(self.Label_VoiceRecordStatus)
        self.h17SpeechControlBox.addWidget(self.StartRecordCommandButton)
        self.h18SpeechControlBox.addWidget(self.Label_WAV_to_CSV)
        self.h19SpeechControlBox.addWidget(self.Label_WAV_to_CSV_Source_Dir)
        self.h19SpeechControlBox.addWidget(self.WAV_to_CSV_Source_Dir)
        self.h19SpeechControlBox.addWidget(self.WAV_to_CSV_Source_Dir_CommandButton)
        self.h20SpeechControlBox.addWidget(self.Label_WAV_to_CSV_Target_Dir)
        self.h20SpeechControlBox.addWidget(self.WAV_to_CSV_Target_Dir)
        self.h20SpeechControlBox.addWidget(self.WAV_to_CSV_Target_Dir_CommandButton)
        self.h21SpeechControlBox.addWidget(self.WAV_to_CSV_Convert_CommandButton)
        self.h21SpeechControlBox.setAlignment(self.WAV_to_CSV_Convert_CommandButton, QtCore.Qt.AlignCenter)
        self.v1SpeechControlBox.addLayout(self.h19SpeechControlBox)
        self.v1SpeechControlBox.addLayout(self.h20SpeechControlBox)
        self.v1SpeechControlBox.addLayout(self.h21SpeechControlBox)
        self.h22SpeechControlBox.addWidget(self.Label_WAV_to_CSV_Convert_Status)
        self.h22SpeechControlBox.addLayout(self.v1SpeechControlBox)
        self.h23SpeechControlBox.addWidget(self.Label_filter_spectrogram)
        self.h24SpeechControlBox.addWidget(self.Label_Filt_Spectr_Dir)
        self.h24SpeechControlBox.addWidget(self.Filt_Spectr_Dir)
        self.h24SpeechControlBox.addWidget(self.Filt_Spectr_Dir_CommandButton)
        self.h25SpeechControlBox.addWidget(self.Label_Filt_Spectr_Status)
        self.h25SpeechControlBox.addWidget(self.Filt_Spectr_CommandButton)
        self.vSpeechControlBox.addWidget(self.Label_VoiceCommand)
        self.vSpeechControlBox.addLayout(self.h1SpeechControlBox)
        self.vSpeechControlBox.addLayout(self.h2SpeechControlBox)
        self.vSpeechControlBox.addLayout(self.h3SpeechControlBox)
        self.vSpeechControlBox.addLayout(self.h4SpeechControlBox)
        self.vSpeechControlBox.addLayout(self.h5SpeechControlBox)
        self.vSpeechControlBox.addLayout(self.h6SpeechControlBox)
        self.vSpeechControlBox.addLayout(self.h7SpeechControlBox)
        self.vSpeechControlBox.addLayout(self.h8SpeechControlBox)
        self.vSpeechControlBox.addLayout(self.h9SpeechControlBox)
        self.vSpeechControlBox.addLayout(self.h10SpeechControlBox)
        self.vSpeechControlBox.addLayout(self.h11SpeechControlBox)
        self.vSpeechControlBox.addLayout(self.h12SpeechControlBox)
        self.vSpeechControlBox.addLayout(self.h13SpeechControlBox)
        self.vSpeechControlBox.addLayout(self.h14SpeechControlBox)
        self.vSpeechControlBox.addLayout(self.h15SpeechControlBox)
        self.vSpeechControlBox.addLayout(self.h16SpeechControlBox)
        self.vSpeechControlBox.addLayout(self.h17SpeechControlBox)
        self.vSpeechControlBox.addLayout(self.h18SpeechControlBox)
        self.vSpeechControlBox.addLayout(self.h19SpeechControlBox)
        self.vSpeechControlBox.addLayout(self.h20SpeechControlBox)
        self.vSpeechControlBox.addLayout(self.h21SpeechControlBox)
        self.vSpeechControlBox.addLayout(self.h22SpeechControlBox)
        self.vSpeechControlBox.addLayout(self.h23SpeechControlBox)
        self.vSpeechControlBox.addLayout(self.h24SpeechControlBox)
        self.vSpeechControlBox.addLayout(self.h25SpeechControlBox)
        self.vSpeechControlBox.addStretch()
        
        self.SpeechControlBox.setLayout(self.vSpeechControlBox)
    
    def delete_file(self):
        try:
            indexItem=self.treeView.selectedIndexes()[0]
        except:
            QtWidgets.QMessageBox.warning(self,"No file is selected.", "Select a file from the tree.")
            return None
        filepath=str(self.treemodel.filePath(indexItem))
        if os.path.isfile(filepath) == False:
            QtWidgets.QMessageBox.warning(self,"No file is selected.", "Select a file from the tree.")
            return None
        reply = QtWidgets.QMessageBox.question(self, "Delete item.", "Confirm to delete?", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel)
        if reply == QtWidgets.QMessageBox.Cancel:
            return None
        else:
            os.remove(filepath)
    
    def plot_stft(self, _filepath):
        if _filepath == False:
            try:
                indexItem=self.treeView.selectedIndexes()[0]
            except:
                QtWidgets.QMessageBox.warning(self,"No file is selected.", "Select a CSV file from the tree.")
                return None
            filepath=str(self.treemodel.filePath(indexItem))
        else:
            filepath = _filepath
        if filepath[-3:] == "csv":
            DA = speech.data_augmentation()
            data, freqseries, timeseries = DA.read_from_csv(filepath)
        else:
            QtWidgets.QMessageBox.warning(self,"Incorrect data file format.", "Select a CSV file from the tree.")
            return None
        try:
            self.subPlt.clear()
            self.canvas.draw()
        except:
            self.subPlt=self.Plt.add_subplot(111, frame_on=False)
            self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
            self.canvas.updateGeometry()
            self.canvas.draw()
        Xlabel="Time, sec"
        Ylabel="Frequency, Hz"
        self.subPlt.set_xlabel(Xlabel)
        self.subPlt.set_ylabel(Ylabel)
        self.subPlt.imshow(data, aspect='auto', origin='lower', extent=[timeseries.min(), timeseries.max(), freqseries.min(), freqseries.max()])
        self.canvas.draw()
    
    def populate_mic_selection(self):
        mic_names_array=[]
        pa = pyaudio.PyAudio()
        info = pa.get_host_api_info_by_index(0)
        devices_n = info.get('deviceCount')
        for i in range(0, devices_n):
            if (pa.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                mic_names_array.append("Input Device id " + str(i) + " - " + pa.get_device_info_by_host_api_device_index(0, i).get('name'))
        self.MicSelectionDropDown.clear()
        self.MicSelectionDropDown.addItems(mic_names_array)
        self.CNNMicSelectionDropDown.clear()
        self.CNNMicSelectionDropDown.addItems(mic_names_array)
        pa = None
        info = None
    
    def volume_bar_update(self, percentage):
        self.mic_levelBar.setValue(percentage)
        self.Label_MicVolume_Value.setText(" " + str(percentage) + " %")
        
    def audio_recording_output_format_manual_change(self):
        global audio_recording_output_format
        if self.Label_RecordToFormatWAV.isChecked():
            audio_recording_output_format = 0
        else:
            audio_recording_output_format = 1
        self.populate_command_list()
    
    def populate_command_list(self):
        global audio_recording_output_format
        directory = QtCore.QDir.currentPath()+'/speech/PERSONAL'
        if audio_recording_output_format == 0:
            directory = str(directory + '/WAV')
        else:
            directory = str(directory + '/CSV')
        folder_names_array = []
        item_names_array = []
        item_names_array = os.listdir(directory)
        for item_name in item_names_array:
            if os.path.isdir(str(directory + '/' + item_name)):
                folder_names_array.append(item_name)
        self.CommandList.clear()
        self.CommandList.addItems(folder_names_array)
    
    def repeats_manual_change(self):
        global command_repeats
        command_repeats = int(str(self.Repeats.currentText()))
    
    def command_duration_manual_change(self):
        global command_duration_manual
        command_duration_manual = self.command_duration_horizontalSlider.value()/1000
        self.command_duration_programmable_change()
    
    def command_duration_programmable_change(self):
        global command_duration
        global command_duration_manual
        global stft_segment_size
        global stft_overlap_percent
        global RATE
        global FREQ
        global CHUNK
        global COMMAND_CHUNK
        n_samples_in_stft_time_step = stft_segment_size - int(stft_segment_size * stft_overlap_percent / 100)
        number_of_time_steps = int(RATE * command_duration_manual / n_samples_in_stft_time_step)
        if abs((number_of_time_steps * n_samples_in_stft_time_step / RATE) - command_duration_manual) > abs(((number_of_time_steps + 1) * n_samples_in_stft_time_step / RATE) - command_duration_manual):
            command_duration = (number_of_time_steps + 1) * n_samples_in_stft_time_step / RATE
        else:
            command_duration = number_of_time_steps * n_samples_in_stft_time_step / RATE
        self.Label_CommandDuration_Value.setText(" " + str(int(command_duration*1000)) + " ms")
        FREQ = RATE/((int(0.05*RATE/stft_segment_size)+1)*stft_segment_size) # number of updates per second
        CHUNK = int(RATE/FREQ) # RATE / number of updates per second
        COMMAND_CHUNK = number_of_time_steps*n_samples_in_stft_time_step-CHUNK
        self.Label_FREQ_Value.setText(str(int(FREQ)) + " Hz")
        self.Label_CHUNK_Value.setText(str(int(CHUNK)))
        self.Label_COMMAND_CHUNK_Value.setText(str(int(COMMAND_CHUNK)))
    
    def RATE_manual_change(self):
        global RATE
        RATE = int(str(self.RATE_Value.currentText()))
        self.command_duration_programmable_change()
        
    def stft_segment_size_manual_change(self):
        global stft_segment_size
        stft_segment_size = self.stft_segment_size_horizontalSlider.value()
        self.command_duration_programmable_change()
        self.Label_SegmentSize_Value.setText(" " + str(stft_segment_size))
        
    def stft_freq_res_manual_change(self):
        global stft_freq_res
        stft_freq_res = self.stft_freq_res_horizontalSlider.value()
        self.Label_stft_freq_res_Value.setText(" " + str(stft_freq_res) + " Hz")
        self.stft_freq_high_lim_programmable_change()
        
    def stft_overlap_percent_manual_change(self):
        global stft_overlap_percent
        stft_overlap_percent = self.stft_overlap_percent_horizontalSlider.value()
        self.Label_stft_overlap_percent_Value.setText(" " + str(stft_overlap_percent) + " %")
        self.command_duration_programmable_change()
        
    def stft_freq_high_lim_manual_change(self):
        global stft_freq_high_lim
        global RATE
        if self.stft_freq_high_lim_horizontalSlider.value() <= RATE * 0.5:
            stft_freq_high_lim = self.stft_freq_high_lim_horizontalSlider.value()
            self.Label_stft_freq_high_lim_Value.setText(" " + str(stft_freq_high_lim) + " Hz")
        else:
            self.stft_freq_high_lim_horizontalSlider.setValue(RATE * 0.5)
            stft_freq_high_lim = self.stft_freq_high_lim_horizontalSlider.value()
            self.Label_stft_freq_high_lim_Value.setText(" " + str(stft_freq_high_lim) + " Hz")
        self.stft_freq_high_lim_programmable_change()
    
    def stft_freq_high_lim_programmable_change(self):
        global stft_freq_high_lim
        global stft_freq_res
        stft_freq_step_size = int(stft_freq_high_lim / stft_freq_res)
        stft_freq_high_lim = stft_freq_res * stft_freq_step_size
        self.Label_stft_freq_high_lim_Value.setText(" " + str(stft_freq_high_lim) + " Hz")
        
    def stft_data_norm_manual_change(self):
        global stft_lin_log_norm
        if self.Label_stft_data_norm_LIN.isChecked():
            stft_lin_log_norm = 0
        else:
            stft_lin_log_norm = 1
            
    def silence_threshold_manual_change(self):
        global silence_treshold_slider
        silence_treshold_slider = self.mic_horizontalSlider.value()
        self.Label_MicThreshold_Value.setText(" " + str(silence_treshold_slider) + " %")
        
    def silence_threshold_programmatic_reset(self, percentage):
        global silence_treshold_slider
        silence_treshold_slider = percentage
        self.mic_horizontalSlider.setValue(percentage)
        self.Label_MicThreshold_Value.setText(" " + str(silence_treshold_slider) + " %")
        self.display_stop_command_recording()
        
    def check_silence_threshold(self):
        global RATE
        global FREQ
        global CHUNK
        
        _signal_message=speech.signal_message()
        _signal_message.progress_update_bit.connect(self.progress_bar_update)
        _signal_message.volume_update_bit.connect(self.volume_bar_update)
        _signal_message.silence_threshold_update_bit.connect(self.silence_threshold_programmatic_reset)
        
        _mic = self.MicSelectionDropDown.currentIndex()
        _rate = RATE
        _freq = FREQ
        _chunk = CHUNK
        _format = pyaudio.paInt16
        _channels = 1
        _sampleduration = 3

        if _mic != "":
             _indentify_silence_threshold = speech.indentify_silence_threshold(_signal_message, _mic, _rate, _freq, _chunk, _format, _channels, _sampleduration)
             _indentify_silence_threshold.start()
             self.display_start_command_recording()
    
    def run_command_recording(self):
        global audio_recording_output_format
        if audio_recording_output_format==0:
            self.recording_to_wav()
        else:
            self.recording_to_csv()
    
    def recording_to_csv(self):
        global RATE
        global CHUNK
        global COMMAND_CHUNK
        global command_repeats
        global silence_treshold_slider
        global stft_segment_size
        global stft_overlap_percent
        global stft_freq_res
        global stft_freq_high_lim
        global stft_lin_log_norm
        
        self._signal_message=speech.signal_message()
        self._signal_message.progress_update_bit.connect(self.progress_bar_update)
        self._signal_message.audio_record_is_ready_bit.connect(self.dialog_record_vocie_sample)
        self._signal_message.audio_recording_completed_bit.connect(self.display_stop_command_recording)
        self._signal_message.volume_update_bit.connect(self.volume_bar_update)
        
        _mic = self.MicSelectionDropDown.currentIndex()
        _rate = RATE
        _chunk = CHUNK
        _commandchunk = COMMAND_CHUNK
        _format = pyaudio.paInt16
        _channels = 1
        _directory = QtCore.QDir.currentPath()+'/speech/PERSONAL/CSV/'+str(self.CommandList.currentText())
        _numberoffiles = command_repeats
        _silencethreshold = silence_treshold_slider
        _stftsegmentsize = stft_segment_size
        _overlappercent = stft_overlap_percent
        _stftfreqres = stft_freq_res
        _freqhighlim = stft_freq_high_lim
        _stft_lin_log_norm = stft_lin_log_norm
        
        if _mic != "":
             _record_audio_csv = speech.record_audio_csv(self._signal_message, _mic, _rate, _chunk, _commandchunk, _format, _channels, _directory, _numberoffiles, _silencethreshold, _stftsegmentsize, _overlappercent, _stftfreqres, _freqhighlim, _stft_lin_log_norm)
             self._signal_message.record_next_audio_bit.connect(_record_audio_csv.change_switch)
             _record_audio_csv.start()
             self.display_start_command_recording()
    
    def recording_to_wav(self):
        global RATE
        global CHUNK
        global COMMAND_CHUNK
        global command_repeats
        global silence_treshold_slider
          
        self._signal_message=speech.signal_message()
        self._signal_message.progress_update_bit.connect(self.progress_bar_update)
        self._signal_message.audio_record_is_ready_bit.connect(self.dialog_record_vocie_sample)
        self._signal_message.audio_recording_completed_bit.connect(self.display_stop_command_recording)
        self._signal_message.volume_update_bit.connect(self.volume_bar_update)
        
        _mic = self.MicSelectionDropDown.currentIndex()
        _rate = RATE
        _chunk = CHUNK
        _commandchunk = COMMAND_CHUNK
        _format = pyaudio.paInt16
        _channels = 1
        _directory = QtCore.QDir.currentPath()+'/speech/PERSONAL/WAV/'+str(self.CommandList.currentText())
        _numberoffiles = command_repeats
        _silencethreshold = silence_treshold_slider
        
        if _mic != "":
             _record_audio_wav = speech.record_audio_wav(self._signal_message, _mic, _rate, _chunk, _commandchunk, _format, _channels, _directory, _numberoffiles, _silencethreshold)
             self._signal_message.record_next_audio_bit.connect(_record_audio_wav.change_switch)
             _record_audio_wav.start()
             self.display_start_command_recording()
        
    def display_start_command_recording(self):
        VoiceRecordStatus = QtGui.QPixmap(QtCore.QDir.currentPath()+'/GUI_images/recording_active.png')
        self.Label_VoiceRecordStatus.setPixmap(VoiceRecordStatus.scaled(22,22))
        self.CheckSilenceCommandButton.setEnabled(False)
            
    def display_stop_command_recording(self):
        VoiceRecordStatus = QtGui.QPixmap(QtCore.QDir.currentPath()+'/GUI_images/recording_inactive.png')
        self.Label_VoiceRecordStatus.setPixmap(VoiceRecordStatus.scaled(22,22))
        self.CheckSilenceCommandButton.setEnabled(True)
        
    def dialog_record_vocie_sample(self, filename, filenumber):
        global command_repeats
        
        if filename != "":
            if filename[-3:] == "csv":
                self.plot_stft(filename)
            reply = QtWidgets.QMessageBox.question(self, "Recording.", "Keep the file?", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            if reply == QtWidgets.QMessageBox.No:
                os.remove(filename)
        if filenumber < command_repeats:
            reply = QtWidgets.QMessageBox.question(self, "Recording.", "Confirm to start recording of voice command <b>'" + str(self.CommandList.currentText()) + "'</b>?", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel)
            if reply == QtWidgets.QMessageBox.Cancel:
                self._signal_message.record_next_audio_bit.emit(-1)
            elif reply == QtWidgets.QMessageBox.Yes:
                self._signal_message.record_next_audio_bit.emit(1)
    
    def WAV_to_CSV_Source_Dir_manual_change(self):
        folder_name = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.WAV_to_CSV_Source_Dir.setText(folder_name)
    
    def WAV_to_CSV_Target_Dir_manual_change(self):
        folder_name = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.WAV_to_CSV_Target_Dir.setText(folder_name)
    
    def converting_wav_to_csv(self):
        _source_directory = self.WAV_to_CSV_Source_Dir.text()
        _target_directory = self.WAV_to_CSV_Target_Dir.text()
        
        if os.path.isdir(_source_directory) == False or os.path.isdir(_target_directory) == False:
            QtWidgets.QMessageBox.warning(self,"Incorrect folder name.", "Specify valid source and target folder names.")
            return None
        
        #clean target directory
        reply = QtWidgets.QMessageBox.question(self, "Conversion of WAV files.", "Confirm to clean target directory?", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            file_names_array = []
            file_names_array = os.listdir(_target_directory)
            if file_names_array != []:
                for file_item in file_names_array:
                    os.remove(str(_target_directory + '/' + file_item))
        
        _signal_message=speech.signal_message()
        _signal_message.progress_update_bit.connect(self.progress_bar_update)
        _signal_message.wav_to_csv_conversion_completed_bit.connect(self.display_WAV_to_CSV_inactive)
        _signal_message.wav_to_csv_spectrogram_is_ready_bit.connect(self.plot_stft)
        
        global stft_segment_size
        global stft_overlap_percent
        global stft_freq_res
        global stft_freq_high_lim
        global stft_lin_log_norm
        global RATE
 
        _stftsegmentsize_ref = stft_segment_size
        _overlappercent = stft_overlap_percent
        _stftfreqres = stft_freq_res
        _freqhighlim = stft_freq_high_lim
        _stft_lin_log_norm = stft_lin_log_norm
        _rate_ref = RATE
        n_samples_in_stft_time_step = stft_segment_size - int(stft_segment_size * stft_overlap_percent / 100)
        _number_of_time_steps = int(RATE * command_duration_manual / n_samples_in_stft_time_step)
        
        _convert_wav_to_csv = speech.convert_wav_to_csv(_signal_message, 
                                                        _source_directory, 
                                                        _target_directory,
                                                        _stftsegmentsize_ref,
                                                        _overlappercent,
                                                        _stftfreqres,
                                                        _freqhighlim,
                                                        _stft_lin_log_norm,
                                                        _rate_ref,
                                                        _number_of_time_steps)
        _convert_wav_to_csv.start()
        
        self.display_WAV_to_CSV_active()
    
    def display_WAV_to_CSV_active(self):
        WAV_to_CSV_Convert_Status = QtGui.QPixmap(QtCore.QDir.currentPath()+'/GUI_images/wav_to_csv_active.png')
        self.Label_WAV_to_CSV_Convert_Status.setPixmap(WAV_to_CSV_Convert_Status.scaled(65,60))
        self.WAV_to_CSV_Convert_CommandButton.setEnabled(False)
            
    def display_WAV_to_CSV_inactive(self):
        WAV_to_CSV_Convert_Status = QtGui.QPixmap(QtCore.QDir.currentPath()+'/GUI_images/wav_to_csv_inactive.png')
        self.Label_WAV_to_CSV_Convert_Status.setPixmap(WAV_to_CSV_Convert_Status.scaled(65,60))
        self.WAV_to_CSV_Convert_CommandButton.setEnabled(True)
    
    def Filt_Spectr_Dir_manual_change(self):
        folder_name = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.Filt_Spectr_Dir.setText(folder_name)
        
    def filter_spectrogram_active(self):
        Filt_Spectr_Status = QtGui.QPixmap(QtCore.QDir.currentPath()+'/GUI_images/filter_active.png')
        self.Label_Filt_Spectr_Status.setPixmap(Filt_Spectr_Status.scaled(42,40))
        self.Filt_Spectr_CommandButton.setEnabled(False)
            
    def filter_spectrogram_inactive(self):
        Filt_Spectr_Status = QtGui.QPixmap(QtCore.QDir.currentPath()+'/GUI_images/filter_inactive.png')
        self.Label_Filt_Spectr_Status.setPixmap(Filt_Spectr_Status.scaled(42,40))
        self.Filt_Spectr_CommandButton.setEnabled(True)
    
    def filter_spectrogram(self):
        _filt_spectr_directory = self.Filt_Spectr_Dir.text()
        self._signal_message = speech.signal_message()
        self._signal_message.progress_update_bit.connect(self.progress_bar_update)
        self._signal_message.spectrogram_filter_completed_bit.connect(self.filter_spectrogram_inactive)
        self._signal_message.spectrogram_filter_abnormal_found_bit.connect(self.filter_spectrogram_dialog)
         
        _spectrogram_filter = speech.spectrogram_filter(self._signal_message, _filt_spectr_directory)
        self._signal_message.spectrogram_filter_action_bit.connect(_spectrogram_filter.change_switch)
        _spectrogram_filter.start()
        
        self.filter_spectrogram_active()
    
    def filter_spectrogram_dialog(self, csv_file_path):
        self.plot_stft(csv_file_path)
        reply = QtWidgets.QMessageBox.question(self, "Spectrogram filtering.", "Abnormal spectrogram is found. Delete the file?", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Abort)
        if reply == QtWidgets.QMessageBox.Yes:
            self._signal_message.spectrogram_filter_action_bit.emit(2)
        elif reply == QtWidgets.QMessageBox.No:
            self._signal_message.spectrogram_filter_action_bit.emit(1)
        elif reply == QtWidgets.QMessageBox.Abort:
            self._signal_message.spectrogram_filter_action_bit.emit(-1)
            
    def CNN_Feed_Dir_manual_change(self):
        folder_name = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.CNN_Feed_Dir.setText(folder_name)
    
    def CNN_Personalize_Dir_manual_change(self):
        folder_name = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.CNN_Personalize_Dir.setText(folder_name)
        
    def personal_data_feed_manual_change(self):
        global personal_data_feed_slider
        personal_data_feed_slider = self.Personal_Use_horizontalSlider.value()
        self.Label_Personal_Use_Value.setText(" " + str(personal_data_feed_slider) + " %")
        
    def CNN_Batches_manual_change(self):
        global CNN_batches
        if self.CNN_Batches.text() != "":
            CNN_batches = int(str(self.CNN_Batches.text()))
        
    def CNN_Epochs_manual_change(self):
        global CNN_epochs
        if self.CNN_Epochs.text() != "":
            CNN_epochs = int(str(self.CNN_Epochs.text()))
        
    def display_CNN_active(self):
        CNN_Status = QtGui.QPixmap(QtCore.QDir.currentPath()+'/GUI_images/CNN_active.png')
        self.Label_CNN_Status.setPixmap(CNN_Status.scaled(65,60))
        self.CNN_Train_CommandButton.setEnabled(False)
            
    def display_CNN_inactive(self):
        CNN_Status = QtGui.QPixmap(QtCore.QDir.currentPath()+'/GUI_images/CNN_inactive.png')
        self.Label_CNN_Status.setPixmap(CNN_Status.scaled(65,60))
        self.CNN_Train_CommandButton.setEnabled(True)
        self.save_CNN_dialog()
        
    def save_CNN_dialog(self):
        reply = QtWidgets.QMessageBox.question(self, "CNN training completed.", "Would you like to save the trained CNN?", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            self._signal_message.cnn_save_action_bit.emit(1)
        else:
            self._signal_message.cnn_save_action_bit.emit(-1)
    
    def save_audio_rec_parameters_for_speech_recognition(self, filename, command_list, accuracy=1):
        global RATE
        global CHUNK
        global COMMAND_CHUNK
        global stft_segment_size
        global stft_overlap_percent
        global stft_freq_res
        global stft_freq_high_lim
        global stft_lin_log_norm
        
        _rate = RATE
        _chunk = CHUNK
        _commandchunk = COMMAND_CHUNK
        _channels = 1
        _stftsegmentsize = stft_segment_size
        _overlappercent = stft_overlap_percent
        _stftfreqres = stft_freq_res
        _freqhighlim = stft_freq_high_lim
        _stft_lin_log_norm = stft_lin_log_norm
        _accuracy = accuracy
        
        with open(filename, 'w', newline='') as csvfile:
            audio_params_csv = csv.writer(csvfile, dialect='excel')
            audio_params_csv.writerow(['_rate',_rate])
            audio_params_csv.writerow(['_chunk',_chunk])
            audio_params_csv.writerow(['_commandchunk',_commandchunk])
            audio_params_csv.writerow(['_channels',_channels])
            audio_params_csv.writerow(['_stftsegmentsize',_stftsegmentsize])
            audio_params_csv.writerow(['_overlappercent',_overlappercent])
            audio_params_csv.writerow(['_stftfreqres',_stftfreqres])
            audio_params_csv.writerow(['_freqhighlim',_freqhighlim])
            audio_params_csv.writerow(['_stft_lin_log_norm',_stft_lin_log_norm])
            command_row = ['command_list']
            for word in command_list:
                command_row.append(word)
            audio_params_csv.writerow(command_row)
            audio_params_csv.writerow(['_accuracy%',_accuracy])
            
            
    def built_CNN(self):
        _CNN_feed_directory = self.CNN_Feed_Dir.text()
        _CNN_personalize_directory = self.CNN_Personalize_Dir.text()
        
        if os.path.isdir(_CNN_feed_directory) == False:
            QtWidgets.QMessageBox.warning(self,"Incorrect CNN feed directory name.", "Specify valid CNN general feed directory name.")
            return None
        
        if os.path.isdir(_CNN_personalize_directory) == False:
            QtWidgets.QMessageBox.warning(self,"Incorrect CNN feed directory name.", "CNN personalize directory name will not be used.")
            _CNN_personalize_directory = ""
        
        self._signal_message=speech.signal_message()
        self._signal_message.cnn_training_over_bit.connect(self.display_CNN_inactive)
        self._signal_message.cnn_saved_name_bit.connect(self.save_audio_rec_parameters_for_speech_recognition)
        
        global personal_data_feed_slider
        global CNN_batches
        global CNN_epochs
        
        _CNN_batches = CNN_batches
        _CNN_epochs = CNN_epochs
        _personalized_percent = personal_data_feed_slider

        SRT = speech.speech_recognition_training(self._signal_message, 
                                       _CNN_feed_directory, 
                                       _CNN_personalize_directory,
                                       _personalized_percent,
                                       _CNN_batches, 
                                       _CNN_epochs)
        
        self._signal_message.cnn_save_action_bit.connect(SRT.change_switch)
        
        SRT.start()
        
        self.display_CNN_active()
    
    def CNN_volume_bar_update(self, percentage):
        self.CNNmic_levelBar.setValue(percentage)
        self.Label_CNNMicVolume_Value.setText(" " + str(percentage) + " %")
    
    def CNN_silence_threshold_manual_change(self):
        global cnn_silence_treshold_slider
        cnn_silence_treshold_slider = self.cnn_mic_horizontalSlider.value()
        self.Label_cnnMicThreshold_Value.setText(" " + str(cnn_silence_treshold_slider) + " %")
        self.CNN_update_mic_and_silence_threshold_in_SRL()
        
    def CNN_silence_threshold_programmatic_reset(self, percentage):
        global cnn_silence_treshold_slider
        cnn_silence_treshold_slider = percentage
        self.cnn_mic_horizontalSlider.setValue(percentage)
        self.Label_cnnMicThreshold_Value.setText(" " + str(cnn_silence_treshold_slider) + " %")
        CNNTestStatus = QtGui.QPixmap(QtCore.QDir.currentPath()+'/GUI_images/CNN_test_inactive.png')
        self.Label_CNNTestStatus.setPixmap(CNNTestStatus.scaled(22,22))
        self.CNN_update_mic_and_silence_threshold_in_SRL()
    
    def CNN_check_silence_threshold(self):
        global RATE
        global FREQ
        global CHUNK
        
        _signal_message=speech.signal_message()
        _signal_message.progress_update_bit.connect(self.progress_bar_update)
        _signal_message.volume_update_bit.connect(self.CNN_volume_bar_update)
        _signal_message.silence_threshold_update_bit.connect(self.CNN_silence_threshold_programmatic_reset)
        
        _mic = self.CNNMicSelectionDropDown.currentIndex()
        _rate = RATE
        _freq = FREQ
        _chunk = CHUNK
        _format = pyaudio.paInt16
        _channels = 1
        _sampleduration = 3

        if _mic != "":
             _indentify_silence_threshold = speech.indentify_silence_threshold(_signal_message, _mic, _rate, _freq, _chunk, _format, _channels, _sampleduration)
             _indentify_silence_threshold.start()
             CNNTestStatus = QtGui.QPixmap(QtCore.QDir.currentPath()+'/GUI_images/CNN_test_active.png')
             self.Label_CNNTestStatus.setPixmap(CNNTestStatus.scaled(22,22))
    
    def CNN_File_manual_change(self):
        directory = str(QtCore.QDir.currentPath() + "/speech/models")
        file_filter = 'HDF5 File (*.hdf5)'
        file_name = str(QtWidgets.QFileDialog.getOpenFileName(self, "Select saved CNN", directory, file_filter))
        file_name = file_name.split("'")[1]
        self.CNN_File.setText(file_name)
        
    def CNN_test_active(self):
        CNNTestStatus = QtGui.QPixmap(QtCore.QDir.currentPath()+'/GUI_images/CNN_test_active.png')
        self.Label_CNNTestStatus.setPixmap(CNNTestStatus.scaled(22,22))
        self.CNN_Test_CommandButton.setText("Stop")
        self.CNN_Load_File_CommandButton.setText("Unload CNN")
        self.CNN_Test_CommandButton.setEnabled(True)
        self.CNN_Load_File_CommandButton.setEnabled(False)
        
    def CNN_test_ready(self):
        CNNTestStatus = QtGui.QPixmap(QtCore.QDir.currentPath()+'/GUI_images/CNN_test_ready.png')
        self.Label_CNNTestStatus.setPixmap(CNNTestStatus.scaled(22,22))
        self.CNN_Test_CommandButton.setText("Test")
        self.CNN_Load_File_CommandButton.setText("Unload CNN")
        self.CNN_Test_CommandButton.setEnabled(True)
        self.CNN_Load_File_CommandButton.setEnabled(True)
        
    def CNN_test_inactive(self):
        CNNTestStatus = QtGui.QPixmap(QtCore.QDir.currentPath()+'/GUI_images/CNN_test_inactive.png')
        self.Label_CNNTestStatus.setPixmap(CNNTestStatus.scaled(22,22))
        self.CNN_Test_CommandButton.setText("Test")
        self.CNN_Load_File_CommandButton.setText("Load CNN")
        self.CNN_Test_CommandButton.setEnabled(False)
        self.CNN_Load_File_CommandButton.setEnabled(True)
        
    def CNN_display_recognized_word(self, recognized_word):
        self.CNN_Result.setText(recognized_word)
        
    def CNN_Load_network(self):
        try:
            if self.SRL:
                self._signal_message.cnn_change_switch_bit.emit(-1)
                del self.SRL
        except:
            _filename_hdf5 = self.CNN_File.text()
            _filename_csv = str(_filename_hdf5[0:len(_filename_hdf5)-4] + "csv")
            if os.path.isfile(_filename_hdf5) == False or os.path.isfile(_filename_csv) == False:
                QtWidgets.QMessageBox.warning(self,"Incorrect CNN file name.", "Specify valid CNN file.")
                return None
            self._signal_message=speech.signal_message()
            self._signal_message.cnn_loaded_bit.connect(self.CNN_test_ready)
            self._signal_message.cnn_unloaded_bit.connect(self.CNN_test_inactive)
            self._signal_message.cnn_waiting_for_speech_bit.connect(self.CNN_test_active)
            self._signal_message.cnn_idle_bit.connect(self.CNN_test_ready)
            self._signal_message.volume_update_bit.connect(self.CNN_volume_bar_update)
            self._signal_message.cnn_spectrogram_is_ready_bit.connect(self.plot_stft)
            self._signal_message.cnn_recognized_bit.connect(self.CNN_display_recognized_word)
            self.SRL = speech.speech_recognition_listening(self._signal_message,
                                                           _filename_hdf5,
                                                           _filename_csv)
            self._signal_message.cnn_change_switch_bit.connect(self.SRL.change_switch)
            self._signal_message.cnn_update_mic_and_silence_threshold_bit.connect(self.SRL.update_mic_and_silence_threshold)
            self.SRL.start()
    
    def CNN_Test_network(self):
        try:
            if self.SRL:
                self.CNN_update_mic_and_silence_threshold_in_SRL()
                self._signal_message.cnn_change_switch_bit.emit(1)
        except:
            QtWidgets.QMessageBox.warning(self,"CNN is not loaded or no microphone is avaliable.", "Load CNN and select microphone.")
            return None
            
    def CNN_update_mic_and_silence_threshold_in_SRL(self):
        global cnn_silence_treshold_slider
        _mic = self.CNNMicSelectionDropDown.currentIndex()
        _silencethreshold = cnn_silence_treshold_slider
        try:
            if self.SRL and _mic != "":
                self._signal_message.cnn_update_mic_and_silence_threshold_bit.emit(_mic, _silencethreshold)
        except:
            return None
    
    def progress_bar_update(self, percentage):
        self.progressBar.setValue(percentage)
        
    def exit_action_custom(self):
        self._quit_signal_message.quit_signal_message_bit.emit(True)
        
    
class quit_signal_message(QtCore.QObject):
    quit_signal_message_bit = QtCore.pyqtSignal(bool)
        
def call(_quit_signal_message):
    speechrecognition_window = SpeechRecognitionWindow(_quit_signal_message)
    speechrecognition_window.show()
    return speechrecognition_window