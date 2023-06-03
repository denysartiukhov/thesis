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

class MainController():
    def __init__(self):
        self.learning_ongoing = False
        self.registration_ongoing = False
        self.mainCamera = cv2.VideoCapture(2)
        self.sideCamera = cv2.VideoCapture(0)
        self.mainCamera.set(3,640)  
        self.mainCamera.set(4,480)
        self.sideCamera.set(3,320)
        self.sideCamera.set(4,240)
        self.root = tkinter.Tk()
        self.root.title('Automated Check-In')
        self.root.configure(background='black', cursor='none')
        self.root.configure(background='black')
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
        #self.viewIdle.mainCameraLabel.place(x=-61, y=0)
        #self.viewIdle.sideCameraLabel.place(x=580, y=120)
        #self.viewIdle.registerNewFaceButton.place(x=635, y=37)
        self.viewIdle.mainCameraLabel.place(x=-61, y=0)
        self.viewIdle.sideCameraLabel.place(x=10, y=120)
        self.viewIdle.registerNewFaceButton.place(x=635, y=37)
        
    def hide_idle(self):
        self.viewIdle.mainCameraLabel.place_forget()
        self.viewIdle.sideCameraLabel.place_forget()
        self.viewIdle.registerNewFaceButton.place_forget()
    
    def draw_registration(self):
        self.viewRegister.sideCameraLabel.place(x=580, y=120)
        self.viewRegister.registrationDoneButton.place(x=284, y=370)
        #self.viewRegister.registrationDoneButton["state"] = "disabled"
        self.viewRegister.registrationCancelButton.place(x=144, y=370)
        self.viewRegister.faceNameText.place(x=30, y=120)
        self.viewRegister.qButton.place(x=30, y=180) 
        self.viewRegister.wButton.place(x=80, y=180)  
        self.viewRegister.eButton.place(x=130, y=180)  
        self.viewRegister.rButton.place(x=180, y=180)  
        self.viewRegister.tButton.place(x=230, y=180)  
        self.viewRegister.yButton.place(x=280, y=180) 
        self.viewRegister.uButton.place(x=330, y=180)
        self.viewRegister.iButton.place(x=380, y=180)  
        self.viewRegister.oButton.place(x=430, y=180)
        self.viewRegister.pButton.place(x=480, y=180)
        self.viewRegister.aButton.place(x=50, y=230) 
        self.viewRegister.sButton.place(x=100, y=230)  
        self.viewRegister.dButton.place(x=150, y=230)  
        self.viewRegister.fButton.place(x=200, y=230)  
        self.viewRegister.gButton.place(x=250, y=230)  
        self.viewRegister.hButton.place(x=300, y=230) 
        self.viewRegister.jButton.place(x=350, y=230)
        self.viewRegister.kButton.place(x=400, y=230)  
        self.viewRegister.lButton.place(x=450, y=230)
        self.viewRegister.zButton.place(x=70, y=280) 
        self.viewRegister.xButton.place(x=120, y=280)  
        self.viewRegister.cButton.place(x=170, y=280)  
        self.viewRegister.vButton.place(x=220, y=280)  
        self.viewRegister.bButton.place(x=270, y=280)  
        self.viewRegister.nButton.place(x=320, y=280) 
        self.viewRegister.mButton.place(x=370, y=280)
        self.viewRegister.backspaceButton.place(x=420, y=280)
        
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
        self.viewRegister.backspaceButton.place_forget()
        
    def stream(self):
        frame_rate = 1
        prev = 0
        while True:
            #logging.debug("debug")
            #logging.info("info")
            time_elapsed = time.time() - prev
            mainImage = self.take_pic(self.mainCamera)
            sideImage = self.take_pic(self.sideCamera)[0:0, 240:240]
            print(sideImage.shape)
            self.display_pic(self.viewIdle.mainCameraLabel,mainImage)
            self.display_pic(self.viewIdle.sideCameraLabel,sideImage)
            self.display_pic(self.viewRegister.sideCameraLabel,sideImage)
            
            if time_elapsed > 1./frame_rate:
                face_names2, face_locations2 = self.find_faces(sideImage)
                if face_names2 != [] and face_names2 != ["Unknown"] and face_names2 and not self.registration_ongoing:
                    face_names1, face_locations1 = self.find_faces(mainImage)
                    if face_names1 != [] and face_names1 != ["Unknown"] and face_names1:
                        if  len(face_names1) == 1 and len(face_names2) == 1 and face_names2 == face_names1:
                            pose1 = self.estimate_pose(sideImage, 1)
                            pose2 = self.estimate_pose(mainImage, 2)
                            self.greet(face_names1[0])
                if self.registration_ongoing and self.learning_ongoing:
                    pose = self.estimate_pose(sideImage, 3)
                    direction = self.estimate_direction(pose) if pose else None
                    if direction == "Straight":
                        cv2.imwrite(f"./test.jpeg", sideImage)
                        self.learn_new_face()
                        self.viewRegister.learningCompletedLabel.place(x=580, y=100)
                        logging.info("Learing complete")
                        self.learning_ongoing = False
                prev = time.time()
                
    def greet(self,name):
        self.hide_idle()
        self.viewIdle.welcomeMessageLabel.config(text=f"Welcome {name}")
        self.viewIdle.welcomeMessageLabel.place(x=0, y=0)
        time.sleep(3)
        self.viewIdle.welcomeMessageLabel.place_forget()
        self.draw_idle()
        
    def on_letter(self,letter):
        self.viewRegister.faceNameText.insert(tkinter.END, letter)
        
    def on_backspace(self,event):
        text = self.viewRegister.faceNameText.get("1.0",tkinter.END)
        text = text[:-2]
        self.viewRegister.faceNameText.delete(f"1.0", tkinter.END)
        self.viewRegister.faceNameText.insert(tkinter.END, text)
        
    def find_faces(self, image):
        sqliteConnection = sqlite3.connect('raspberry.db')
        cursor = sqliteConnection.cursor()
        encodings, names = self.model.getEncodings(cursor)
        encodings = np.array(encodings)
        sqliteConnection.close()
        face_locations = []
        face_encodings = []
        face_names = []
             
        small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
                             
                    # Find all the faces and face encodings in the current frame of video

        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        face_names = []
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

            # Draw a box around the face
            #cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            #cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            #font = cv2.FONT_HERSHEY_DUPLEX
            #cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        if not face_names == []:
            return face_names[0], face_locations
        else:
            return None, None
        
    def estimate_pose(self,some_image, index):
        some_image.flags.writeable = False
        if index == 1:
            results = self.model.face_mesh1.process(some_image)
        elif index == 2:
            results = self.model.face_mesh2.process(some_image)
        else:
            results = self.model.face_mesh3.process(some_image)
        some_image.flags.writeable = True
        img_h, img_w, img_c = some_image.shape

        face_3d = []
        face_2d = []
        x = 0
        y = 0

        if results.multi_face_landmarks:
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
        if float(pose_estimate[0]) > -15 and float(pose_estimate[0]) < 15 and float(pose_estimate[1]) > -10 and float(pose_estimate[1]) < 10:
            return "Straight"
        elif float(pose_estimate[0]) > -15 and float(pose_estimate[0]) < 15 and float(pose_estimate[1]) < -10:
            return "Right"
        elif float(pose_estimate[0]) > -15 and float(pose_estimate[0]) < 15 and float(pose_estimate[1]) > 10:
            return "Left"
    
    def learn_new_face(self):
        obama_image = face_recognition.load_image_file("./test.jpeg")
        obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
        self.model.new_face_encoding_temp = [obama_face_encoding]
        
    def take_pic(self,cam):
        success, image = cam.read()
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def display_pic(self,label,image):
        image1 = Image.fromarray(image)
        frame_image = ImageTk.PhotoImage(image1)
        label.config(image=frame_image)
        label.image = frame_image