import sys
import os
import json
import argparse
from PyQt5.QtWidgets import (
    QApplication, QListWidget, QListWidgetItem, QWidget, QTextEdit, QLabel, QVBoxLayout, QHBoxLayout, QShortcut)
from PyQt5 import QtGui
from PyQt5.QtCore import QSize

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        with open(args.json_file, 'r') as f:
            self.data = json.load(f)
        with open(args.query_file, 'r') as f:
            self.query = json.load(f)
        self.initUI()
    
    def initUI(self):
        self.data_query = list(self.data.keys())
        self.first_query_id = self.data_query[0]

        self.nl_query = QTextEdit()
        self.nl_query.setReadOnly(True)
        nl_text = self.query[self.first_query_id]
        self.nl_query.append('\n'.join(nl_text))
        self.nl_query.setFixedSize(QSize(300,100))

        self.list_query = QListWidget()
        for query_id in self.data_query:
            self.list_query.addItem(query_id)
        self.list_query.clicked.connect(self.query_clicked)

        self.list_track = QListWidget()
        for track_id in self.data[self.first_query_id]:
            self.list_track.addItem(track_id)
        self.list_track.clicked.connect(self.track_clicked)

        self.list_frame = QListWidget()
        self.list_frame.setViewMode(QListWidget.IconMode)
        root_frame = os.path.join("tracks","test",self.data[self.first_query_id][0])
        image_list = os.listdir(root_frame)
        image_list.sort()
        for image in image_list:
            item = QListWidgetItem()
            icon = QtGui.QIcon()
            pixmap = QtGui.QPixmap(os.path.join(root_frame,image))
            pixmap = pixmap.scaledToHeight(500)
            icon.addPixmap(pixmap)
            item.setIcon(icon)
            self.list_frame.setIconSize(QSize(500,500))
            self.list_frame.addItem(item)

        vbox_1st = QVBoxLayout()
        self.label_nl = QLabel('Query Description', self)
        self.label_nl.setFixedWidth(300)
        vbox_1st.addWidget(self.label_nl)
        vbox_1st.addWidget(self.nl_query)
        self.label_query = QLabel('Query ID', self)
        self.label_query.setFixedWidth(300)
        vbox_1st.addWidget(self.label_query)
        vbox_1st.addWidget(self.list_query)

        vbox_2nd = QVBoxLayout()
        self.label_track = QLabel('Track ID', self)
        self.label_track.setFixedWidth(300)
        vbox_2nd.addWidget(self.label_track)
        vbox_2nd.addWidget(self.list_track)

        self.vbox_3rd = QVBoxLayout()
        self.label_frame = QLabel('Frames', self)
        self.label_frame.setMinimumHeight(50)
        self.vbox_3rd.addWidget(self.label_frame)
        self.vbox_3rd.addWidget(self.list_frame)

        hbox = QHBoxLayout()
        hbox.addLayout(vbox_1st)
        hbox.addLayout(vbox_2nd)
        hbox.addLayout(self.vbox_3rd)

        self.setLayout(hbox)
        self.setGeometry(300, 300, 1000, 500)
        self.show()

    def query_clicked(self):
        self.list_track.clear()
        query_id = self.list_query.currentItem().text()
        for track_id in self.data[query_id]:
            self.list_track.addItem(track_id)
        self.nl_query.clear()
        nl_text = self.query[query_id]
        self.nl_query.append('\n'.join(nl_text))
            

    def track_clicked(self):
        self.list_frame.clear()
        track_id = self.list_track.currentItem().text()
        root_frame = os.path.join("tracks","test",track_id)
        image_list = os.listdir(root_frame)
        image_list.sort()
        for image in image_list:
            item = QListWidgetItem()
            icon = QtGui.QIcon()
            pixmap = QtGui.QPixmap(os.path.join(root_frame,image))
            pixmap = pixmap.scaledToHeight(500)
            icon.addPixmap(pixmap)
            item.setIcon(icon)
            self.list_frame.setIconSize(QSize(500,500))
            self.list_frame.addItem(item)

def main(args):
    app = QApplication(sys.argv)
    myapp = MyApp()
    sys.exit(app.exec_())

if __name__ == '__main__':
    print("Loading parameters...")
    parser = argparse.ArgumentParser(description='Visualize test result')
    parser.add_argument('--json-file', dest='json_file', default='baseline/baseline-results.json')
    parser.add_argument('--query-file', dest='query_file', default='data/test-queries.json')

    args = parser.parse_args()

    main(args)
