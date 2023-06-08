import sys
import os
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

sys.path.append(f'{os.getcwd()}/src')
from raspberry_stereo.views.ViewIdle import ViewIdle
from raspberry_stereo.views.ViewRegister import ViewRegister
from raspberry_stereo.models.MainModel import MainModel

folder = "Denys_JPEGs_color"
#light = "bright"
light = "half-bright"
#light = "half-dark"
#light = "ultra-dark"
distance = "close"
pose = "straight"
person = "D"


class MainController():
    def __init__(self, args):
        self.args = args
        self.learning_ongoing = False
        self.registration_ongoing = False
        self.main_camera = cv2.VideoCapture(2)
        self.side_camera = cv2.VideoCapture(0)
        self.main_camera.set(3,640)  
        self.main_camera.set(4,480)
        self.side_camera.set(3,320)
        self.side_camera.set(4,240)
        self.main_image = None
        self.side_image = None
        self.register_image = None
        self.root = tkinter.Tk()
        self.root.title('Automated Check-In')
        self.root.configure(background='black', cursor='none')
        self.root.geometry("800x480")
        self.root.attributes('-fullscreen', True)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.model = MainModel()
        self.view_idle = ViewIdle(self.root)
        self.view_register = ViewRegister(self.root)
        self.view_idle.register_new_face_button.bind('<Button>', self.start_registration)
        self.view_register.registration_done_button.bind('<Button>', self.complete_registration)
        self.view_register.registration_cancel_button.bind('<Button>', self.exit_registration)
        self.view_register.a_button.bind('<Button>', lambda event: self.on_letter('A'))
        self.view_register.b_button.bind('<Button>', lambda event: self.on_letter('B'))
        self.view_register.c_button.bind('<Button>', lambda event: self.on_letter('C'))
        self.view_register.d_button.bind('<Button>', lambda event: self.on_letter('D'))
        self.view_register.e_button.bind('<Button>', lambda event: self.on_letter('E'))
        self.view_register.f_button.bind('<Button>', lambda event: self.on_letter('F'))
        self.view_register.g_button.bind('<Button>', lambda event: self.on_letter('G'))
        self.view_register.h_button.bind('<Button>', lambda event: self.on_letter('H'))
        self.view_register.i_button.bind('<Button>', lambda event: self.on_letter('I'))
        self.view_register.j_button.bind('<Button>', lambda event: self.on_letter('J'))
        self.view_register.k_button.bind('<Button>', lambda event: self.on_letter('K'))
        self.view_register.l_button.bind('<Button>', lambda event: self.on_letter('L'))
        self.view_register.m_button.bind('<Button>', lambda event: self.on_letter('M'))
        self.view_register.n_button.bind('<Button>', lambda event: self.on_letter('N'))
        self.view_register.o_button.bind('<Button>', lambda event: self.on_letter('O'))
        self.view_register.p_button.bind('<Button>', lambda event: self.on_letter('P'))
        self.view_register.q_button.bind('<Button>', lambda event: self.on_letter('Q'))
        self.view_register.r_button.bind('<Button>', lambda event: self.on_letter('R'))
        self.view_register.s_button.bind('<Button>', lambda event: self.on_letter('S'))
        self.view_register.t_button.bind('<Button>', lambda event: self.on_letter('T'))
        self.view_register.u_button.bind('<Button>', lambda event: self.on_letter('U'))
        self.view_register.v_button.bind('<Button>', lambda event: self.on_letter('V'))
        self.view_register.w_button.bind('<Button>', lambda event: self.on_letter('W'))
        self.view_register.x_button.bind('<Button>', lambda event: self.on_letter('X'))
        self.view_register.y_button.bind('<Button>', lambda event: self.on_letter('Y'))
        self.view_register.z_button.bind('<Button>', lambda event: self.on_letter('Z'))
        self.view_register.zero_button.bind('<Button>', lambda event: self.on_letter('0'))
        self.view_register.one_button.bind('<Button>', lambda event: self.on_letter('1'))
        self.view_register.two_button.bind('<Button>', lambda event: self.on_letter('2'))
        self.view_register.three_button.bind('<Button>', lambda event: self.on_letter('3'))
        self.view_register.four_button.bind('<Button>', lambda event: self.on_letter('4'))
        self.view_register.five_button.bind('<Button>', lambda event: self.on_letter('5'))
        self.view_register.six_button.bind('<Button>', lambda event: self.on_letter('6'))
        self.view_register.seven_button.bind('<Button>', lambda event: self.on_letter('7'))
        self.view_register.eight_button.bind('<Button>', lambda event: self.on_letter('8'))
        self.view_register.nine_button.bind('<Button>', lambda event: self.on_letter('9'))
        self.view_register.backspace_button.bind('<Button>', self.on_backspace)
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
        text = self.view_register.face_name_text.get("1.0",tkinter.END)
        self.model.save_encoding(text, self.model.new_face_encoding_temp[0])
        self.model.new_face_encoding_temp = []
        self.view_register.face_name_text.delete("1.0", tkinter.END)
        self.exit_registration(event)
        
    def exit_registration(self, event):
        self.view_register.face_name_text.delete("1.0", tkinter.END)
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
        self.view_idle.already_checked_in_label.place(x=0, y=101)
        self.view_idle.main_camera_label.place(x=240, y=0)
        self.view_idle.side_camera_label.place(x=0, y=120)
        self.view_idle.register_new_face_button.place(x=65, y=37)
        self.view_idle.checked_in_list_label.place(x=610, y=40)
        self.view_idle.checked_in_list_header_label.place(x=610, y=0)
        
    def hide_idle(self):
        self.view_idle.already_checked_in_label.place_forget()
        self.view_idle.main_camera_label.place_forget()
        self.view_idle.side_camera_label.place_forget()
        self.view_idle.register_new_face_button.place_forget()
        self.view_idle.checked_in_list_label.place_forget()
        self.view_idle.checked_in_list_header_label.place_forget()
    
    def draw_registration(self):
        self.view_register.side_camera_label.place(x=0, y=120)
        self.view_register.registration_done_button.place(x=524, y=370)
        self.view_register.registration_cancel_button.place(x=384, y=370)
        self.view_register.face_name_text.place(x=270, y=120)
        self.view_register.q_button.place(x=270, y=210) 
        self.view_register.w_button.place(x=320, y=210)  
        self.view_register.e_button.place(x=370, y=210)  
        self.view_register.r_button.place(x=420, y=210)  
        self.view_register.t_button.place(x=470, y=210)  
        self.view_register.y_button.place(x=520, y=210) 
        self.view_register.u_button.place(x=570, y=210)
        self.view_register.i_button.place(x=620, y=210)  
        self.view_register.o_button.place(x=670, y=210)
        self.view_register.p_button.place(x=720, y=210)
        self.view_register.a_button.place(x=290, y=260) 
        self.view_register.s_button.place(x=340, y=260)  
        self.view_register.d_button.place(x=390, y=260)  
        self.view_register.f_button.place(x=440, y=260)  
        self.view_register.g_button.place(x=490, y=260)  
        self.view_register.h_button.place(x=540, y=260) 
        self.view_register.j_button.place(x=590, y=260)
        self.view_register.k_button.place(x=640, y=260)  
        self.view_register.l_button.place(x=690, y=260)
        self.view_register.z_button.place(x=310, y=310) 
        self.view_register.x_button.place(x=360, y=310)  
        self.view_register.c_button.place(x=410, y=310)  
        self.view_register.v_button.place(x=460, y=310)  
        self.view_register.b_button.place(x=510, y=310)  
        self.view_register.n_button.place(x=560, y=310) 
        self.view_register.m_button.place(x=610, y=310)
        self.view_register.zero_button.place(x=270, y=160)
        self.view_register.one_button.place(x=320, y=160)
        self.view_register.two_button.place(x=370, y=160)
        self.view_register.three_button.place(x=420, y=160)
        self.view_register.four_button.place(x=470, y=160)
        self.view_register.five_button.place(x=520, y=160)
        self.view_register.six_button.place(x=570, y=160)
        self.view_register.seven_button.place(x=620, y=160)
        self.view_register.eight_button.place(x=670, y=160)
        self.view_register.nine_button.place(x=720, y=160)
        self.view_register.backspace_button.place(x=660, y=310)
        
    def hide_registration(self):
        self.view_register.learning_completed_label.place_forget()
        self.view_register.side_camera_label.place_forget()
        self.view_register.registration_done_button.place_forget()
        self.view_register.registration_cancel_button.place_forget()
        self.view_register.face_name_text.place_forget()
        self.view_register.q_button.place_forget() 
        self.view_register.w_button.place_forget()  
        self.view_register.e_button.place_forget()
        self.view_register.r_button.place_forget() 
        self.view_register.t_button.place_forget() 
        self.view_register.y_button.place_forget()
        self.view_register.u_button.place_forget()
        self.view_register.i_button.place_forget()  
        self.view_register.o_button.place_forget()
        self.view_register.p_button.place_forget()
        self.view_register.a_button.place_forget()
        self.view_register.s_button.place_forget()
        self.view_register.d_button.place_forget()
        self.view_register.f_button.place_forget()
        self.view_register.g_button.place_forget()
        self.view_register.h_button.place_forget()
        self.view_register.j_button.place_forget()
        self.view_register.k_button.place_forget()
        self.view_register.l_button.place_forget()
        self.view_register.z_button.place_forget()
        self.view_register.x_button.place_forget()
        self.view_register.c_button.place_forget() 
        self.view_register.v_button.place_forget()
        self.view_register.b_button.place_forget()
        self.view_register.n_button.place_forget()
        self.view_register.m_button.place_forget()
        self.view_register.zero_button.place_forget()
        self.view_register.one_button.place_forget()
        self.view_register.two_button.place_forget()
        self.view_register.three_button.place_forget()
        self.view_register.four_button.place_forget()
        self.view_register.five_button.place_forget()
        self.view_register.six_button.place_forget()
        self.view_register.seven_button.place_forget()
        self.view_register.eight_button.place_forget()
        self.view_register.nine_button.place_forget()
        self.view_register.backspace_button.place_forget()
        
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
                    face_names_side, face_locations_side = self.find_faces(self.side_image)
                    if face_names_side != [] and face_names_side != ["Unknown"] and face_names_side:
                        logging.debug(f"Face found in the side frame.")
                        logging.debug(f"Looking for faces in the main frame...")
                        face_names_main, face_locations_main = self.find_faces(self.main_image)
                        if face_names_main != [] and face_names_main != ["Unknown"] and face_names_main:
                            if  len(face_names_main) == 1 and face_names_side == face_names_main:
                                logging.debug(f"Face found in the main frame.")
                                pose_side = self.estimate_pose(self.side_image, 1)
                                pose_main = self.estimate_pose(self.main_image, 2)
                                angle_difference = None
                                try:
                                    angle_difference = float(pose_side[1]) - float(pose_main[1])
                                    logging.debug(f"The angle difference between side frame and main frame is {angle_difference}.")
                                except:
                                    logging.debug(f"Cannot estimate angle difference between side frame and main frame.")
                                user_first_name = self.model.get_user_info(face_names_main[0])
                                if self.model.is_checked_in(face_names_main[0]):
                                    self.view_idle.already_checked_in_label.config(text=f"{user_first_name} already checked id.", bg="green")
                                    timer = 1
                                elif angle_difference != None and angle_difference > 10:
                                    logging.debug(f"{user_first_name} has not been checked id yet. Checking in...")
                                    self.model.check_in(face_names_main[0])
                                    self.greet(user_first_name)
                if self.registration_ongoing and self.learning_ongoing:
                    pose = self.estimate_pose(self.register_image, 2)
                    direction = self.estimate_direction(pose) if pose else None
                    if direction == "Straight":
                        self.learn_new_face()
                        self.view_register.learning_completed_label.place(x=0, y=100)
                        logging.info("Learning completed.")
                        self.learning_ongoing = False
                prev = time.time()

                if timer > 0 and timer < 10:
                    timer += 1
                else:
                    timer = 0
                    self.view_idle.already_checked_in_label.config(text="", bg="black")

                
    def greet(self,name):
        self.hide_idle()
        self.view_idle.welcome_message_label.config(text=f"Welcome {name}")
        self.view_idle.welcome_message_label.place(x=0, y=0)
        time.sleep(2)
        self.view_idle.welcome_message_label.place_forget()
        self.draw_idle()

    def update_checked_in_list(self):
        user_name_list = self.model.get_checked_in_users()
        self.view_idle.checked_in_list_label.config(text=user_name_list)
        
    def on_letter(self,letter):
        self.view_register.face_name_text.insert(tkinter.END, letter)
        
    def on_backspace(self,event):
        text = self.view_register.face_name_text.get("1.0",tkinter.END)
        text = text[:-2]
        self.view_register.face_name_text.delete(f"1.0", tkinter.END)
        self.view_register.face_name_text.insert(tkinter.END, text)
        
    def find_faces(self, image):
        encodings, names = self.model.get_encodings()
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
        face_encodings = face_recognition.face_encodings(self.register_image)[0]
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
            self.main_image = self.take_pic(self.main_camera,480,yMain)
            self.side_image = self.take_pic(self.side_camera,240,240)
            self.register_image = self.side_image
        else:
            #self.main_image = cv2.imread(f"/home/dartiukhov/Desktop/thesis_clean/thesis/{folder}/set2/{light}_color/{distance}_{pose}2_c.jpg")
            #self.side_image = cv2.imread(f"/home/dartiukhov/Desktop/thesis_clean/thesis/{folder}/set2/{light}_color/{distance}_{pose}1_c.jpg")
            self.main_image = cv2.imread(f"/home/dartiukhov/Desktop/thesis_clean/thesis/{folder}/set1/{person}_{light}_c.jpg")
            self.side_image = cv2.imread(f"/home/dartiukhov/Desktop/thesis_clean/thesis/{folder}/set1/{person}_{light}_c.jpg")
            self.register_image = cv2.imread(f"/home/dartiukhov/Desktop/thesis_clean/thesis/{folder}/set1/{person}_{light}_c.jpg")
        self.display_pic(self.view_idle.main_camera_label,self.main_image)
        self.display_pic(self.view_idle.side_camera_label,self.side_image)
        self.display_pic(self.view_register.side_camera_label,self.register_image)

    def display_pic(self,label,image):
        image1 = Image.fromarray(image)
        frame_image = ImageTk.PhotoImage(image1)
        label.config(image=frame_image)
        label.image = frame_image