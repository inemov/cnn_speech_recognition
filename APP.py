# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:09:46 2020
@author: Ivan Nemov
"""
from PyQt5 import QtWidgets
import gui_speech
import sys

speech_GUI_window = None

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.form_widget = FormWidget(self)
        _widget = QtWidgets.QWidget()
        _layout = QtWidgets.QVBoxLayout(_widget)
        _layout.addWidget(self.form_widget)
        self.setCentralWidget(_widget)
        self.showMinimized()
        self.setWindowTitle("")
        _quit_signal_message=gui_speech.quit_signal_message()
        _quit_signal_message.quit_signal_message_bit.connect(self.quit_function)
        global speech_GUI_window
        speech_GUI_window = gui_speech.call(_quit_signal_message)
        
    def quit_function(self):
        global speech_GUI_window
        speech_GUI_window = None
        
class FormWidget(QtWidgets.QWidget):
    
    def __init__(self, parent):
        super(FormWidget, self).__init__(parent)
        self.__controls()
        self.__layout()

    def __controls(self):
        
        self.menu_bar=QtWidgets.QMenuBar()
        file_menu=self.menu_bar.addMenu("File")
        exit_action=QtWidgets.QAction('Exit',self)
        exit_action.triggered.connect(QtWidgets.QApplication.quit)
        file_menu.addAction(exit_action)
        
    def __layout(self):
        return None
        
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    app.exec_()
    
if __name__ == '__main__':
    sys.exit(main())