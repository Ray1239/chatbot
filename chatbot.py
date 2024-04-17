from PyQt5 import QtCore, QtGui, QtWidgets
from dotenv import load_dotenv
import os
import textwrap
import numpy as np
import pandas as pd
import PyPDF2
import google.generativeai as genai
import google.ai.generativelanguage as glm
import sys

load_dotenv()
GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
aimodel = genai.GenerativeModel('models/gemini-pro')
model = 'models/embedding-001'

def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""

        for page_num in range(len(pdf_reader.pages)):
            # Extract text from the current page
            page = pdf_reader.pages[page_num]
            text += page.extract_text()

        # Close the PDF file
        file.close()

        # Return the extracted text
        return text
    
def convertToChunks(chunk_size, text):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
    return chunks

def embed_query(query):
    request = genai.embed_content(model=model, content=query, task_type="retrieval_query")
    return request

def embed_fn(text):
    return genai.embed_content(model=model,
                                content=text,
                                task_type="retrieval_document")["embedding"]

def applyEmbedToDF(df):
    df['Embeddings'] = df.apply(lambda row: embed_fn(row['Text']), axis=1)

def find_best_passage(query, dataframe):
    """
    Compute the distances between the query and each document in the dataframe
    using the dot product.
    """
    query_embedding = genai.embed_content(model=model,
                                            content=query,
                                            task_type="retrieval_query")
    dot_products = np.dot(np.stack(dataframe['Embeddings']), query_embedding["embedding"])
    idx = np.argmax(dot_products)
    return dataframe.iloc[idx]['Text'] # Return text from index with max value

def make_prompt(query, relevant_passage):
    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = textwrap.dedent("""You are a helpful and informative bot that answers questions using text from the reference passage included below. \
    Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
    However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
    strike a friendly and converstional tone. \
    QUESTION: '{query}'
    PASSAGE: '{relevant_passage}'

    ANSWER:
    """).format(query=query, relevant_passage=escaped)

    return prompt

class EmbeddingThread(QtCore.QThread):
    finished = QtCore.pyqtSignal(pd.DataFrame)
    def __init__(self, file_path, df):
        super().__init__()
        self.file_path = file_path
        self.df = df
    def run(self):
        text = read_pdf(file_path=self.file_path)
        doc = convertToChunks(500, text=text)

        if hasattr(self, 'df') and isinstance(self.df, pd.DataFrame) and not self.df.empty:
            # If dataframe already exists, append new rows
            new_df = pd.DataFrame({'Text': doc})
            applyEmbedToDF(new_df)
            self.df = pd.concat([self.df, new_df], ignore_index=True)
            print(self.df)
        else:
            # If dataframe does not exist, create a new one
            self.df = pd.DataFrame({'Text': doc})
            applyEmbedToDF(self.df)
        self.finished.emit(self.df)

class QueryThread(QtCore.QThread):
    finished = QtCore.pyqtSignal(str)

    def __init__(self, query, dataframe):
        super().__init__()
        self.query = query
        self.dataframe = dataframe

    def run(self):
        passage = find_best_passage(self.query, self.dataframe)
        prompt = make_prompt(self.query, passage)
        answer = aimodel.generate_content(prompt)
        response = f"You: {self.query}<br><br>Answer: {answer.text}"
        self.finished.emit(response)

class LoadingWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        layout = QtWidgets.QHBoxLayout()
        layout.setAlignment(QtCore.Qt.AlignCenter) 
        self.loadingLabel = QtWidgets.QLabel()
        movie = QtGui.QMovie("loading.gif")
        movie.setScaledSize(QtCore.QSize(50, 50)) 
        self.loadingLabel.setMovie(movie)
        movie.start()
        layout.addWidget(self.loadingLabel)
        self.setLayout(layout)

class MainWindowL(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.selectFile.clicked.connect(self.openFileDialog)
        self.ui.sendQuery.clicked.connect(self.sendQuery)
        self.ui.queryInput.installEventFilter(self)
        self.df = None

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.KeyPress and obj is self.ui.queryInput:
            if event.key() == QtCore.Qt.Key_Return and event.modifiers() & QtCore.Qt.ShiftModifier:
                cursor = self.ui.queryInput.textCursor()
                cursor.insertText('\n')
                return True
            elif event.key() == QtCore.Qt.Key_Return:
                if self.ui.queryInput.toPlainText().strip() != "":
                    self.sendQuery()
                return True
        return super().eventFilter(obj, event)
    
    def openFileDialog(self):
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            print("Selected file:", fileName)
            item = QtWidgets.QListWidgetItem(os.path.basename(fileName))  # Create a list item with the file name
            self.ui.fileList.addItem(item)
            self.progress = QtWidgets.QProgressDialog("Embedding...", "Cancel", 0, 0, self)
            self.progress.setWindowModality(QtCore.Qt.WindowModal)
            self.progress.setMinimumDuration(0)
            self.progress.show()

            self.embedding_thread = EmbeddingThread(file_path=fileName, df=self.df)
            self.embedding_thread.finished.connect(self.embeddingFinished)
            self.embedding_thread.start()
    def embeddingFinished(self, df):
        self.df = df
        self.progress.cancel()

    def sendQuery(self):
        text = self.ui.queryInput.toPlainText()
        self.ui.queryInput.clear()

        # Show loading icon
        item = QtWidgets.QListWidgetItem()
        loading_widget = LoadingWidget()
        self.ui.listWidget_2.addItem(item)
        self.ui.listWidget_2.setItemWidget(item, loading_widget)
        item.setSizeHint(loading_widget.sizeHint())

        # Create and start query thread
        self.query_thread = QueryThread(query=text, dataframe=self.df)
        self.query_thread.finished.connect(lambda response: self.showResponse(item, response))
        self.query_thread.finished.connect(self.query_thread.deleteLater) 
        self.query_thread.start()

    def showResponse(self, item, response):
        # Replace loading icon with response
        label = QtWidgets.QLabel()
        label.setText(response)
        label.setStyleSheet("background-color: grey; border: 1px solid black; border-radius: 10px; padding: 5px; color: white; width: 100%;")
        label.setWordWrap(True)
        self.ui.listWidget_2.setItemWidget(item, label)
        item.setSizeHint(label.sizeHint())
        

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1800, 1200)
        MainWindow.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.sideWidget = QtWidgets.QWidget(parent=self.centralwidget)
        self.sideWidget.setGeometry(QtCore.QRect(40, 0, 300, 1151))
        self.sideWidget.setObjectName("sideWidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(parent=self.sideWidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 0, 300, 1151))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        spacerItem = QtWidgets.QSpacerItem(20, 110, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        self.verticalLayout.addItem(spacerItem)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)

        #select File button
        self.selectFileLabel = QtWidgets.QLabel(parent=self.verticalLayoutWidget)
        self.selectFileLabel.setObjectName("selectFileLabel")
        self.horizontalLayout.addWidget(self.selectFileLabel)
        self.selectFile = QtWidgets.QPushButton(parent=self.verticalLayoutWidget)
        self.selectFile.setMinimumSize(QtCore.QSize(60, 0))
        self.selectFile.setMaximumSize(QtCore.QSize(120, 16777215))
        self.selectFile.setText("")
        # icon = QtGui.QIcon.fromTheme("folder-open")
        # self.selectFile.setIcon(icon)
        icon_name = "folder-open"  # Name of the icon you want to check
        if QtGui.QIcon.hasThemeIcon(icon_name):
            icon = QtGui.QIcon.fromTheme(icon_name)
            self.selectFile.setIcon(icon)
        else:
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap("open-folder.svg"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
            self.selectFile.setIcon(icon)

        # spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        # self.horizontalLayout.addItem(spacerItem2)
        self.selectFile.setObjectName("selectFile")
        self.horizontalLayout.addWidget(self.selectFile)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout.addItem(spacerItem2)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setContentsMargins(-1, 20, -1, -1)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem3)


        #files list widget
        self.fileList = QtWidgets.QListWidget(parent=self.verticalLayoutWidget)
        self.fileList.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.fileList.setStyleSheet("#listWidget{\n"
        "    border: 1px solid black;\n"
        "    border-radius: 10px;\n"
        "    background-color: white;\n"
        "}")
        self.fileList.setObjectName("fileList")
        self.horizontalLayout_7.addWidget(self.fileList)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem4)
        self.verticalLayout.addLayout(self.horizontalLayout_7)
        spacerItem5 = QtWidgets.QSpacerItem(20, 80, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout.addItem(spacerItem5)
        self.mainWidget = QtWidgets.QWidget(parent=self.centralwidget)
        self.mainWidget.setGeometry(QtCore.QRect(250, 0, 1551, 1161))
        self.mainWidget.setStyleSheet("")
        self.mainWidget.setObjectName("mainWidget")

        #chat area
        self.chatArea = QtWidgets.QScrollArea(parent=self.mainWidget)
        self.chatArea.setGeometry(QtCore.QRect(90, 100, 1391, 731))
        self.chatArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.chatArea.setStyleSheet("#chatArea{\n"
        "    padding: 5px;\n"
        "    padding-top: 15px;\n"
        "    padding-bottom: 15px;\n"
        "    border: 1px solid transparent;\n"
        "}")
        self.chatArea.setWidgetResizable(True)
        self.chatArea.setObjectName("chatArea")
        self.chatAreaContents = QtWidgets.QWidget()
        self.chatAreaContents.setGeometry(QtCore.QRect(0, 0, 1359, 699))
        self.chatAreaContents.setMaximumWidth(1359)
        self.chatAreaContents.setStyleSheet("\n"
        "#listWidget_2{\n"
        "    background-color: white;\n"
        "}")
        self.chatAreaContents.setObjectName("chatAreaContents")
        self.horizontalLayoutWidget_3 = QtWidgets.QWidget(parent=self.chatAreaContents)
        self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(70, 0, 1251, 691))
        self.horizontalLayoutWidget_3.setObjectName("horizontalLayoutWidget_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem6 = QtWidgets.QSpacerItem(50, 20, QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem6)

        #answer list
        self.listWidget_2 = QtWidgets.QListWidget(parent=self.horizontalLayoutWidget_3)
        self.listWidget_2.setStyleSheet("#listWidget_2{\n"
        "    border: 1px solid black;\n"
        "    border-radius: 10px;\n"
        "    background-color: white;\n"
        "}"
        "#listWidget_2 QAbstractItemView::item { border-top: 1px solid black;   border-bottom: 1px solid black; padding: 10px; }")
        self.listWidget_2.setObjectName("listWidget_2")
        self.listWidget_2.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.listWidget_2.setMaximumWidth(1159)
        self.horizontalLayout_3.addWidget(self.listWidget_2)
        spacerItem7 = QtWidgets.QSpacerItem(50, 20, QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem7)
        self.chatArea.setWidget(self.chatAreaContents)
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(parent=self.mainWidget)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(340, 830, 891, 88))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        spacerItem8 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout_2.addItem(spacerItem8)

        #query area
        self.queryInput = QtWidgets.QTextEdit(parent=self.horizontalLayoutWidget_2)
        self.queryInput.setMaximumSize(QtCore.QSize(16777215, 50))
        self.queryInput.setStyleSheet("#queryInput{\n"
        "    border: 1px solid rgb(94, 92, 100);\n"
        "    border-radius: 10px;\n"
        "    background-color: white;\n"
        "}")
        self.queryInput.setObjectName("queryInput")
        self.verticalLayout_2.addWidget(self.queryInput)
        spacerItem9 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout_2.addItem(spacerItem9)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.sendQuery = QtWidgets.QPushButton(parent=self.horizontalLayoutWidget_2)
        self.sendQuery.setMinimumSize(QtCore.QSize(50, 50))
        self.sendQuery.setMaximumSize(QtCore.QSize(50, 50))
        self.sendQuery.setStyleSheet("#sendQuery{\n"
        "    background-color: white;\n"
        "    border: 1px solid  black;\n"
        "    border-radius: 15px;\n"
        "}")
        self.sendQuery.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("send.svg"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.sendQuery.setIcon(icon)
        self.sendQuery.setObjectName("sendQuery")
        self.horizontalLayout_2.addWidget(self.sendQuery)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1800, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Chat With Documents"))
        self.selectFileLabel.setText(_translate("MainWindow", "Select File: "))
        self.queryInput.setPlaceholderText(_translate("MainWindow", "Ask me queries about your documents"))

    class CustomWidget(QtWidgets.QWidget):
        def __init__(self, text):
            super().__init__()
            layout = QtWidgets.QVBoxLayout()
            self.label = QtWidgets.QLabel(text)
            self.label.setStyleSheet("border: 1px solid black; padding: 5px;")
            layout.addWidget(self.label)
            self.setLayout(layout)

    class MyListWidget(QtWidgets.QListWidget):
        def __init__(self):
            super().__init__()
            self.setStyleSheet("QListWidget { border: 1px solid black; padding: 5px; }")

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    # MainWindow = QtWidgets.QMainWindow()
    MainWindow = MainWindowL()
    MainWindow.show()
    sys.exit(app.exec())
