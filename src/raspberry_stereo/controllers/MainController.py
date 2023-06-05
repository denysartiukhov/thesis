import sys
import os
sys.path.append(f'{os.getcwd()}/src')
from raspberry_stereo.views.ViewIdle import ViewIdle
from raspberry_stereo.views.ViewRegister import ViewRegister
from raspberry_stereo.models.MainModel import MainModel
import logging
import sqlite3
import tkinter, threading
import PIL
from PIL import ImageTk
from PIL import Image
import sys
import cv2
import time
import face_recognition
import mediapipe as mp
import numpy as np

folder = "Denys_JPEGs_color"
#light = "bright"
light = "half-bright"
#light = "half-dark"
#light = "ultra-dark"
distance = "far"
pose = "left"
person = "D"


class MainController():
    def __init__(self, args):
        self.args = args
        self.learning_ongoing = False
        self.registration_ongoing = False
        self.mainCamera = cv2.VideoCapture(2)
        self.sideCamera = cv2.VideoCapture(0)
        self.mainCamera.set(3,640)  
        self.mainCamera.set(4,480)
        self.sideCamera.set(3,320)
        self.sideCamera.set(4,240)
        self.mainImage = None
        self.sideImage = None
        self.registerImage = None
        self.root = tkinter.Tk()
        self.root.title('Automated Check-In')
        self.root.configure(background='black', cursor='none')
        self.root.geometry("800x480")
        self.root.attributes('-fullscreen', True)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.model = MainModel()
        self.viewIdle = ViewIdle(self.root)
        self.viewRegister = ViewRegister(self.root)
        self.viewIdle.registerNewFaceButton.bind('<Button>', self.start_registration)
        self.viewRegister.registrationDoneButton.bind('<Button>', self.complete_registration)
        self.viewRegister.registrationCancelButton.bind('<Button>', self.exit_registration)
        self.viewRegister.aButton.bind('<Button>', lambda event: self.on_letter('A'))
        self.viewRegister.bButton.bind('<Button>', lambda event: self.on_letter('B'))
        self.viewRegister.cButton.bind('<Button>', lambda event: self.on_letter('C'))
        self.viewRegister.dButton.bind('<Button>', lambda event: self.on_letter('D'))
        self.viewRegister.eButton.bind('<Button>', lambda event: self.on_letter('E'))
        self.viewRegister.fButton.bind('<Button>', lambda event: self.on_letter('F'))
        self.viewRegister.gButton.bind('<Button>', lambda event: self.on_letter('G'))
        self.viewRegister.hButton.bind('<Button>', lambda event: self.on_letter('H'))
        self.viewRegister.iButton.bind('<Button>', lambda event: self.on_letter('I'))
        self.viewRegister.jButton.bind('<Button>', lambda event: self.on_letter('J'))
        self.viewRegister.kButton.bind('<Button>', lambda event: self.on_letter('K'))
        self.viewRegister.lButton.bind('<Button>', lambda event: self.on_letter('L'))
        self.viewRegister.mButton.bind('<Button>', lambda event: self.on_letter('M'))
        self.viewRegister.nButton.bind('<Button>', lambda event: self.on_letter('N'))
        self.viewRegister.oButton.bind('<Button>', lambda event: self.on_letter('O'))
        self.viewRegister.pButton.bind('<Button>', lambda event: self.on_letter('P'))
        self.viewRegister.qButton.bind('<Button>', lambda event: self.on_letter('Q'))
        self.viewRegister.rButton.bind('<Button>', lambda event: self.on_letter('R'))
        self.viewRegister.sButton.bind('<Button>', lambda event: self.on_letter('S'))
        self.viewRegister.tButton.bind('<Button>', lambda event: self.on_letter('T'))
        self.viewRegister.uButton.bind('<Button>', lambda event: self.on_letter('U'))
        self.viewRegister.vButton.bind('<Button>', lambda event: self.on_letter('V'))
        self.viewRegister.wButton.bind('<Button>', lambda event: self.on_letter('W'))
        self.viewRegister.xButton.bind('<Button>', lambda event: self.on_letter('X'))
        self.viewRegister.yButton.bind('<Button>', lambda event: self.on_letter('Y'))
        self.viewRegister.zButton.bind('<Button>', lambda event: self.on_letter('Z'))
        self.viewRegister.zeroButton.bind('<Button>', lambda event: self.on_letter('0'))
        self.viewRegister.oneButton.bind('<Button>', lambda event: self.on_letter('1'))
        self.viewRegister.twoButton.bind('<Button>', lambda event: self.on_letter('2'))
        self.viewRegister.threeButton.bind('<Button>', lambda event: self.on_letter('3'))
        self.viewRegister.fourButton.bind('<Button>', lambda event: self.on_letter('4'))
        self.viewRegister.fiveButton.bind('<Button>', lambda event: self.on_letter('5'))
        self.viewRegister.sixButton.bind('<Button>', lambda event: self.on_letter('6'))
        self.viewRegister.sevenButton.bind('<Button>', lambda event: self.on_letter('7'))
        self.viewRegister.eightButton.bind('<Button>', lambda event: self.on_letter('8'))
        self.viewRegister.nineButton.bind('<Button>', lambda event: self.on_letter('9'))
        self.viewRegister.backspaceButton.bind('<Button>', self.on_backspace)
        self.thread = threading.Thread(target=self.stream)
        self.thread.daemon = 1
        self.draw_idle()

        
          
    def run(self):
        self.thread.start()
        self.root.mainloop()
    
    def close(self):
        self.root.destroy()
        sys.exit()
    
    def complete_registration(self, event):
        text = self.viewRegister.faceNameText.get("1.0",tkinter.END)
        self.model.saveEncoding(text, self.model.new_face_encoding_temp[0])
        self.model.new_face_encoding_temp = []
        self.viewRegister.faceNameText.delete("1.0", tkinter.END)
        self.exit_registration(event)
        
    def exit_registration(self, event):
        self.viewRegister.faceNameText.delete("1.0", tkinter.END)
        self.registration_ongoing = False
        self.learning_ongoing = False
        self.hide_registration()
        self.draw_idle()
    
    def start_registration(self, event):
        self.learning_ongoing = True
        self.registration_ongoing = True
        self.hide_idle()
        self.draw_registration()
    
    def draw_idle(self):
        self.viewIdle.alreadyCheckedInLabel.place(x=0, y=101)
        self.viewIdle.mainCameraLabel.place(x=240, y=0)
        self.viewIdle.sideCameraLabel.place(x=0, y=120)
        self.viewIdle.registerNewFaceButton.place(x=65, y=37)
        self.viewIdle.checkedInList.place(x=620, y=0)
        
    def hide_idle(self):
        self.viewIdle.alreadyCheckedInLabel.place_forget()
        self.viewIdle.mainCameraLabel.place_forget()
        self.viewIdle.sideCameraLabel.place_forget()
        self.viewIdle.registerNewFaceButton.place_forget()
        self.viewIdle.checkedInList.place_forget()
    
    def draw_registration(self):
        self.viewRegister.sideCameraLabel.place(x=0, y=120)
        self.viewRegister.registrationDoneButton.place(x=524, y=370)
        self.viewRegister.registrationCancelButton.place(x=384, y=370)
        self.viewRegister.faceNameText.place(x=270, y=120)
        self.viewRegister.qButton.place(x=270, y=210) 
        self.viewRegister.wButton.place(x=320, y=210)  
        self.viewRegister.eButton.place(x=370, y=210)  
        self.viewRegister.rButton.place(x=420, y=210)  
        self.viewRegister.tButton.place(x=470, y=210)  
        self.viewRegister.yButton.place(x=520, y=210) 
        self.viewRegister.uButton.place(x=570, y=210)
        self.viewRegister.iButton.place(x=620, y=210)  
        self.viewRegister.oButton.place(x=670, y=210)
        self.viewRegister.pButton.place(x=720, y=210)
        self.viewRegister.aButton.place(x=290, y=260) 
        self.viewRegister.sButton.place(x=340, y=260)  
        self.viewRegister.dButton.place(x=390, y=260)  
        self.viewRegister.fButton.place(x=440, y=260)  
        self.viewRegister.gButton.place(x=490, y=260)  
        self.viewRegister.hButton.place(x=540, y=260) 
        self.viewRegister.jButton.place(x=590, y=260)
        self.viewRegister.kButton.place(x=640, y=260)  
        self.viewRegister.lButton.place(x=690, y=260)
        self.viewRegister.zButton.place(x=310, y=310) 
        self.viewRegister.xButton.place(x=360, y=310)  
        self.viewRegister.cButton.place(x=410, y=310)  
        self.viewRegister.vButton.place(x=460, y=310)  
        self.viewRegister.bButton.place(x=510, y=310)  
        self.viewRegister.nButton.place(x=560, y=310) 
        self.viewRegister.mButton.place(x=610, y=310)
        self.viewRegister.zeroButton.place(x=270, y=160)
        self.viewRegister.oneButton.place(x=320, y=160)
        self.viewRegister.twoButton.place(x=370, y=160)
        self.viewRegister.threeButton.place(x=420, y=160)
        self.viewRegister.fourButton.place(x=470, y=160)
        self.viewRegister.fiveButton.place(x=520, y=160)
        self.viewRegister.sixButton.place(x=570, y=160)
        self.viewRegister.sevenButton.place(x=620, y=160)
        self.viewRegister.eightButton.place(x=670, y=160)
        self.viewRegister.nineButton.place(x=720, y=160)
        self.viewRegister.backspaceButton.place(x=660, y=310)
        
    def hide_registration(self):
        self.viewRegister.learningCompletedLabel.place_forget()
        self.viewRegister.sideCameraLabel.place_forget()
        self.viewRegister.registrationDoneButton.place_forget()
        self.viewRegister.registrationCancelButton.place_forget()
        self.viewRegister.faceNameText.place_forget()
        self.viewRegister.qButton.place_forget() 
        self.viewRegister.wButton.place_forget()  
        self.viewRegister.eButton.place_forget()
        self.viewRegister.rButton.place_forget() 
        self.viewRegister.tButton.place_forget() 
        self.viewRegister.yButton.place_forget()
        self.viewRegister.uButton.place_forget()
        self.viewRegister.iButton.place_forget()  
        self.viewRegister.oButton.place_forget()
        self.viewRegister.pButton.place_forget()
        self.viewRegister.aButton.place_forget()
        self.viewRegister.sButton.place_forget()
        self.viewRegister.dButton.place_forget()
        self.viewRegister.fButton.place_forget()
        self.viewRegister.gButton.place_forget()
        self.viewRegister.hButton.place_forget()
        self.viewRegister.jButton.place_forget()
        self.viewRegister.kButton.place_forget()
        self.viewRegister.lButton.place_forget()
        self.viewRegister.zButton.place_forget()
        self.viewRegister.xButton.place_forget()
        self.viewRegister.cButton.place_forget() 
        self.viewRegister.vButton.place_forget()
        self.viewRegister.bButton.place_forget()
        self.viewRegister.nButton.place_forget()
        self.viewRegister.mButton.place_forget()
        self.viewRegister.zeroButton.place_forget()
        self.viewRegister.oneButton.place_forget()
        self.viewRegister.twoButton.place_forget()
        self.viewRegister.threeButton.place_forget()
        self.viewRegister.fourButton.place_forget()
        self.viewRegister.fiveButton.place_forget()
        self.viewRegister.sixButton.place_forget()
        self.viewRegister.sevenButton.place_forget()
        self.viewRegister.eightButton.place_forget()
        self.viewRegister.nineButton.place_forget()
        self.viewRegister.backspaceButton.place_forget()
        
    def stream(self):
        frame_rate = 1
        prev = 0
        timer = 0
        while True:
            time_elapsed = time.time() - prev

            self.capture_frames()
            self.update_checked_in_list()
            
            if time_elapsed > 1./frame_rate:
                if not self.registration_ongoing:
                    logging.debug(f"Looking for faces in the side frame...")
                    face_names_side, face_locations_side = self.find_faces(self.sideImage)
                    if face_names_side != [] and face_names_side != ["Unknown"] and face_names_side:
                        logging.debug(f"Face found in the side frame.")
                        logging.debug(f"Looking for faces in the main frame...")
                        face_names_main, face_locations_main = self.find_faces(self.mainImage)
                        if face_names_main != [] and face_names_main != ["Unknown"] and face_names_main:
                            if  len(face_names_main) == 1 and face_names_side == face_names_main:
                                logging.debug(f"Face found in the main frame.")
                                pose_side = self.estimate_pose(self.sideImage, 1)
                                pose_main = self.estimate_pose(self.mainImage, 2)
                                try:
                                    angle_difference = float(pose_side[1]) - float(pose_main[1])
                                    logging.debug(f"The angle difference between side frame and main frame is {angle_difference}.")
                                except:
                                    logging.debug(f"Cannot estimate angle difference between side frame and main frame.")
                                user_first_name = self.model.getUserInfo(face_names_main[0])
                                if self.model.isCheckedIn(face_names_main[0]):
                                    self.viewIdle.alreadyCheckedInLabel.config(text=f"{user_first_name} already checked id.", bg="green")
                                    timer = 1
                                elif angle_difference > 10:
                                    self.viewIdle.alreadyCheckedInLabel.config(text=f"{user_first_name} has not been checked id yet. Checking in...", bg="green")
                                    self.model.checkIn(face_names_main[0])
                                    self.greet(user_first_name)
                if self.registration_ongoing and self.learning_ongoing:
                    pose = self.estimate_pose(self.registerImage, 2)
                    direction = self.estimate_direction(pose) if pose else None
                    if direction == "Straight":
                        self.learn_new_face()
                        self.viewRegister.learningCompletedLabel.place(x=0, y=100)
                        logging.info("Learning completed.")
                        self.learning_ongoing = False
                prev = time.time()

                if timer > 0 and timer < 10:
                    timer += 1
                else:
                    timer = 0
                    self.viewIdle.alreadyCheckedInLabel.config(text="", bg="black")

                
    def greet(self,name):
        self.hide_idle()
        self.viewIdle.welcomeMessageLabel.config(text=f"Welcome {name}")
        self.viewIdle.welcomeMessageLabel.place(x=0, y=0)
        time.sleep(2)
        self.viewIdle.welcomeMessageLabel.place_forget()
        self.draw_idle()

    def update_checked_in_list(self):
        user_name_list = self.model.get_checked_in_users()
        self.viewIdle.checkedInList.config(text=user_name_list)
        
    def on_letter(self,letter):
        self.viewRegister.faceNameText.insert(tkinter.END, letter)
        
    def on_backspace(self,event):
        text = self.viewRegister.faceNameText.get("1.0",tkinter.END)
        text = text[:-2]
        self.viewRegister.faceNameText.delete(f"1.0", tkinter.END)
        self.viewRegister.faceNameText.insert(tkinter.END, text)
        
    def find_faces(self, image):
        encodings, names = self.model.getEncodings()
        encodings = np.array(encodings)
        face_locations = []
        face_encodings = []
        face_names = []
  
        small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
                             
        # Find all the faces and face encodings in the current frame of video

        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(encodings, face_encoding)
            name = "Unknown"
            # Use the known face with the smallest distance to the new face
            if not encodings == []:
                face_distances = face_recognition.face_distance(encodings, face_encoding)
                try:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = names[best_match_index]
                except:
                    pass

            face_names.append(name)
        
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

        if not face_names == []:
            return face_names[0], face_locations
        else:
            return None, None
        
    def estimate_pose(self,some_image, index):
        results = None
        some_image.flags.writeable = False
        try:
            if index == 1:
                results = self.model.face_mesh1.process(some_image)
            elif index == 2:
                results = self.model.face_mesh2.process(some_image)
            else:
                results = self.model.face_mesh3.process(some_image)
        except Exception as e:
            pass
        some_image.flags.writeable = True
        img_h, img_w, img_c = some_image.shape

        face_3d = []
        face_2d = []
        x = 0
        y = 0

        if results and results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])       
                
                # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                        [0, focal_length, img_w / 2],
                                        [0, 0, 1]])

                # The Distance Matrix
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360
                x = '%.3f'%(x)
                y = '%.3f'%(y)

                return tuple([x, y])
    
    def estimate_direction(self,pose_estimate):
        if float(pose_estimate[0]) > -20 and float(pose_estimate[0]) < 20 and float(pose_estimate[1]) > -20 and float(pose_estimate[1]) < 20:
            return "Straight"
        elif float(pose_estimate[0]) > -20 and float(pose_estimate[0]) < 20 and float(pose_estimate[1]) < -10:
            return "Right"
        elif float(pose_estimate[0]) > -20 and float(pose_estimate[0]) < 20 and float(pose_estimate[1]) > 10:
            return "Left"
    
    def learn_new_face(self):
        face_encodings = face_recognition.face_encodings(self.registerImage)[0]
        self.model.new_face_encoding_temp = [face_encodings]
        
    def take_pic(self,cam,x,y):
        success, image = cam.read()
        image = image[0:x, 0:y]
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image
    
    def capture_frames(self):
        yMain = 560 if logging.root.level == logging.INFO else 360
        if not self.args.train_from_source:
            self.mainImage = self.take_pic(self.mainCamera,480,yMain)
            self.sideImage = self.take_pic(self.sideCamera,240,240)
            self.registerImage = self.sideImage
        else:
            self.mainImage = cv2.imread(f"/home/dartiukhov/Desktop/thesis_clean/thesis/{folder}/set2/{light}_color/{distance}_{pose}2_c.jpg")
            self.sideImage = cv2.imread(f"/home/dartiukhov/Desktop/thesis_clean/thesis/{folder}/set2/{light}_color/{distance}_{pose}1_c.jpg")
            self.registerImage = cv2.imread(f"/home/dartiukhov/Desktop/thesis_clean/thesis/{folder}/set1/{person}_{light}_c.jpg")
        self.display_pic(self.viewIdle.mainCameraLabel,self.mainImage)
        self.display_pic(self.viewIdle.sideCameraLabel,self.sideImage)
        self.display_pic(self.viewRegister.sideCameraLabel,self.registerImage)

    def display_pic(self,label,image):
        image1 = Image.fromarray(image)
        frame_image = ImageTk.PhotoImage(image1)
        label.config(image=frame_image)
        label.image = frame_image