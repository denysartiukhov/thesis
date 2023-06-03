import tkinter, threading
import PIL
from PIL import ImageTk
from PIL import Image
import cv2
import time
import face_recognition
import mediapipe as mp
import numpy as np
import sqlite3
import logging

class MainModel():
    def __init__(self):
        self.new_face_encoding_temp = []
        self.known_face_encodings = []
        self.known_face_names = []
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh1 = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.face_mesh2 = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.face_mesh3 = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        self.sqliteConnection = sqlite3.connect('raspberry.db')
        self.cursor = self.sqliteConnection.cursor()

        rows = self.cursor.fetchall()
        if len(rows) == 0:
            self.cursor.executescript("CREATE TABLE faces (id TEXT PRIMARY KEY, encoding TEXT, first_name TEXT NOT NULL, last_name TEXT NOT NULL, creation_date DATETIME DEFAULT CURRENT_TIMESTAMP)")
            self.cursor.executescript("CREATE TABLE checked_in (id TEXT PRIMARY KEY, check_in_date DATETIME DEFAULT CURRENT_TIMESTAMP)")

            self.cursor.executescript(f"INSERT INTO faces(id, first_name, last_name) VALUES('BISERA', 'Bisera', 'Nestorovska')")
            self.cursor.executescript(f"INSERT INTO faces(id, first_name, last_name) VALUES('DENYS', 'Denys', 'Artiukhov')")

            logging.debug("Face encodings table does not exist, creating...")
        else:
            logging.debug("Face encodings table already exists, nothing to do.")

        
    def saveEncoding(self, id, encoding):
        self.cursor.executescript(f"INSERT INTO faces(id, encoding) VALUES('{id}', '{encoding}')")
        logging.debug(f"Encoding for {id} saved to DB.")
    
    def isCheckedIn(self, id):
        sqliteConnection = sqlite3.connect('raspberry.db')
        cursor = sqliteConnection.cursor()
        cursor.execute(f"SELECT * FROM checked_in WHERE id = '{id}';")
        rows = cursor.fetchall()
        sqliteConnection.close()
        if len(rows) == 0:
            print('Checking in...')
            return False
        else:
            print('Already checked in')
            return True

    def checkIn(self, id):
        sqliteConnection = sqlite3.connect('raspberry.db')
        cursor = sqliteConnection.cursor()
        cursor.executescript(f"INSERT INTO checked_in(id) VALUES('{id}')")
        sqliteConnection.close()
        logging.debug(f"User {id} checked in for the day.")
        
    def getEncodings(self, cursor):
        cursor.execute(f"SELECT * FROM faces;")
        rows = cursor.fetchall()
        encodings, ids = [], []
        if len(rows) == 0:
            return [], []
        for row in rows:
            encodingsList = row[1].replace('[','').replace(']','').replace('\n','').split(' ')
            encodingsList = [float(x) for x in encodingsList if x != '']
            idsList = row[0].replace('[','').replace(']','').replace('\n','').split(' ')
            idsList = [x for x in idsList if x != '']
            encodings.append(encodingsList)
            ids.append(idsList)
        logging.debug(f"Encodings fetched from DB.")
        return encodings, ids
        