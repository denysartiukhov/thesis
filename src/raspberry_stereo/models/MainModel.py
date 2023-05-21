import tkinter, threading
import PIL
from PIL import ImageTk
from PIL import Image
import cv2
import time
import face_recognition
import mediapipe as mp
import numpy as np

class Model():
    def __init__(self):
        self.new_face_encoding_temp = []
        self.known_face_encodings = []
        self.known_face_names = []
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh1 = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.face_mesh2 = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)