import tkinter, threading
import PIL
from PIL import ImageTk
from PIL import Image
import cv2
import mediapipe as mp
import numpy as np
import time
import face_recognition
import sys

global registrationOngoing
global learning_complete
global known_face_enconings
global known_face_names
global new_face_temp
global new_face_name_temp
new_face_encoding_temp = []
new_face_name_temp = []
known_face_encodings = []
known_face_names = []
registrationOngoing = False
learning_complete = False
mp_face_mesh = mp.solutions.face_mesh
face_mesh1 = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh2 = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def find_faces(image):
    face_locations = []
    face_encodings = []
    face_names = []
         
    small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
                         
                # Find all the faces and face encodings in the current frame of video

    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        # Use the known face with the smallest distance to the new face
        if not known_face_encodings == []:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        face_names.append(name)
    
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    if not face_names == []:
        return face_names, face_locations
    else:
        return None, None

def learn_new_face():
    global known_face_names
    global known_face_encodings
    global new_face_encoding_temp
    obama_image = face_recognition.load_image_file("./images/test.jpeg")
    obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
    new_face_encoding_temp = [obama_face_encoding]

        
def take_pic(cam):
    success, image = cam.read()
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def display_pic(label,image):
    image1 = Image.fromarray(image)
    frame_image = ImageTk.PhotoImage(image1)
    label.config(image=frame_image)
    label.image = frame_image
        
def estimate_direction(pose_estimate):
    print(pose_estimate)
    if float(pose_estimate[0]) > -15 and float(pose_estimate[0]) < 15 and float(pose_estimate[1]) > -10 and float(pose_estimate[1]) < 10:
        return "Straight"
    elif float(pose_estimate[0]) > -10 and float(pose_estimate[0]) < 10 and float(pose_estimate[1]) < -4:
        return "Right"
    elif float(pose_estimate[0]) > -10 and float(pose_estimate[0]) < 10 and float(pose_estimate[1]) > 4:
        return "Left"
    
def estimate_pose(some_image, index):
    some_image.flags.writeable = False
    if index == 1:
        results = face_mesh1.process(some_image)
    else:
        results = face_mesh2.process(some_image)
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

def stream(label1,mainCamera,label2,sideCamera):
    camera = 1
    global learning_complete
    global registrationOngoing
    global known_face_names
    global known_face_encodings
    learning_complete = False
    i = 0
    frame_rate = 1
    prev = 0
    known_face_names = []
    known_face_encodings = []

    while True:
        time_elapsed = time.time() - prev 
        image1 = take_pic(mainCamera)
        display_pic(label1,image1)
        image2 = take_pic(sideCamera)
        display_pic(label2,image2)  
        if time_elapsed > 1./frame_rate:
                face_names2, face_locations2 = find_faces(image2)
                if face_names2 != [] and face_names2 != ["Unknown"] and face_names2 and not registrationOngoing:
                    face_names1, face_locations1 = find_faces(image1)
                    if face_names1 != [] and face_names1 != ["Unknown"] and face_names1:
                        if  len(face_names1) == 1 and len(face_names2) == 1 and face_names2 == face_names1:
                            greet(face_names1[0])
                    pose1 = estimate_pose(image1, 1)
                    pose2 = estimate_pose(image2, 2)
                if registrationOngoing and learning_complete == False:
                    pose = estimate_pose(image2, 1)
                    direction = estimate_direction(pose) if pose else None
                    if direction == "Straight":
                        cv2.imwrite(f"./images/test.jpeg", image2)
                        learn_new_face()
                        learningCompletedLabel.place(x=580, y=120)
                        print("Learing complete")
                        learning_complete = True
                prev = time.time()

def register():
    global registrationOngoing
    registrationOngoing = True
    destroy_screen()
    
    draw_registration_screen()
    
def greet(name):
    destroy_screen()
    welcomeMessageLabel.config(text=f"Welcome {name}")
    welcomeMessageLabel.place(x=0, y=0)
    time.sleep(3)
    welcomeMessageLabel.place_forget()
    draw_main_screen()
    
def done():
    global learning_complete
    global known_face_names
    global known_face_encodings
    global registrationOngoing
    global new_face_encoding_temp
    registrationOngoing = False
    learning_complete = False 
    text = faceNameText.get("1.0",tkinter.END)
    known_face_names.append(text)
    known_face_encodings.append(new_face_encoding_temp[0])
    new_face_encoding_temp = []
    faceNameText.delete("1.0", tkinter.END)
    destroy_screen()
    draw_main_screen()
    
def on_registration_cancel():
    global learning_complete
    global known_face_names
    global known_face_encodings
    global registrationOngoing
    global new_face_encoding_temp
    registrationOngoing = False
    learning_complete = False 
    new_face_encoding_temp = []
    faceNameText.delete("1.0", tkinter.END)
    destroy_screen()
    draw_main_screen()
    
      
def on_letter(letter):
    faceNameText.insert(tkinter.END, letter)
    # Label Creation
    
def on_backspace():
    text = faceNameText.get("1.0",tkinter.END)
    text = text[:-2]
    faceNameText.delete(f"1.0", tkinter.END)
    faceNameText.insert(tkinter.END, text)
    
    
def draw_registration_screen():
    sideCameraLabel.place(x=580, y=120)
    registrationDoneButton.place(x=284, y=370)
    registrationCancelButton.place(x=144, y=370)
    faceNameText.place(x=30, y=120)
    qButton.place(x=30, y=180) 
    wButton.place(x=80, y=180)  
    eButton.place(x=130, y=180)  
    rButton.place(x=180, y=180)  
    tButton.place(x=230, y=180)  
    yButton.place(x=280, y=180) 
    uButton.place(x=330, y=180)
    iButton.place(x=380, y=180)  
    oButton.place(x=430, y=180)
    pButton.place(x=480, y=180)
    aButton.place(x=50, y=230) 
    sButton.place(x=100, y=230)  
    dButton.place(x=150, y=230)  
    fButton.place(x=200, y=230)  
    gButton.place(x=250, y=230)  
    hButton.place(x=300, y=230) 
    jButton.place(x=350, y=230)
    kButton.place(x=400, y=230)  
    lButton.place(x=450, y=230)
    zButton.place(x=70, y=280) 
    xButton.place(x=120, y=280)  
    cButton.place(x=170, y=280)  
    vButton.place(x=220, y=280)  
    bButton.place(x=270, y=280)  
    nButton.place(x=320, y=280) 
    mButton.place(x=370, y=280)
    backspaceButton.place(x=420, y=280)
    
def draw_main_screen():
    mainCameraLabel.place(x=-61, y=0)
    sideCameraLabel.place(x=580, y=120)
    registerNewFaceButton.place(x=635, y=37)

    
def destroy_screen():
    registerNewFaceButton.place_forget()
    registrationDoneButton.place_forget()
    registrationCancelButton.place_forget()
    faceNameText.place_forget()
    qButton.place_forget()
    wButton.place_forget()
    eButton.place_forget()
    rButton.place_forget()
    tButton.place_forget()
    yButton.place_forget()
    uButton.place_forget()
    iButton.place_forget()
    oButton.place_forget()
    pButton.place_forget()
    aButton.place_forget()
    sButton.place_forget()
    dButton.place_forget()
    fButton.place_forget()
    gButton.place_forget()
    hButton.place_forget()
    jButton.place_forget()
    kButton.place_forget()
    lButton.place_forget()
    zButton.place_forget()
    xButton.place_forget()
    cButton.place_forget()
    vButton.place_forget()
    bButton.place_forget()
    nButton.place_forget()
    mButton.place_forget()
    backspaceButton.place_forget()
    mainCameraLabel.place_forget()
    learningCompletedLabel.place_forget()
    sideCameraLabel.place_forget()
    welcomeMessageLabel.place_forget()
    

def on_closing():
    root.destroy()
    sys.exit()


if __name__ == "__main__":

    mainCamera = cv2.VideoCapture(2)
    sideCamera = cv2.VideoCapture(0)
    mainCamera.set(3,640)  
    mainCamera.set(4,480)
    sideCamera.set(3,320)
    sideCamera.set(4,240)

    root = tkinter.Tk()
    root.title('Automated Check-In')

    root.configure(background='black', cursor='none')
    root.geometry("800x480")
    root.attributes('-fullscreen', True)
    root.protocol("WM_DELETE_WINDOW", on_closing)

    mainCameraLabel = tkinter.Label(root, borderwidth=0)
    sideCameraLabel = tkinter.Label(root, borderwidth=0)
    learningCompletedLabel = tkinter.Label(root, text="Learning completed.", borderwidth=0, bg="green", fg="white")
    welcomeMessageLabel = tkinter.Label(root, text="", borderwidth=0, height=12, width=38, background='green',font=("tahoma",25), fg="white")

    registerNewFaceButton = tkinter.Button(root, text='Register', width=8, command=register, relief="flat", bg="white", font="tahoma")
    registrationDoneButton = tkinter.Button(root, text='Done', width=10, command=done, relief="flat", bg="white", font="tahoma")
    registrationCancelButton = tkinter.Button(root, text='Cancel', width=10, command=on_registration_cancel, relief="flat", bg="white", font="tahoma")

    aButton = tkinter.Button(root, text='A', width=1, command=lambda: on_letter('A'), relief="flat", bg="white", font="tahoma")
    bButton = tkinter.Button(root, text='B', width=1, command=lambda: on_letter('B'), relief="flat", bg="white", font="tahoma")
    cButton = tkinter.Button(root, text='C', width=1, command=lambda: on_letter('C'), relief="flat", bg="white", font="tahoma")
    dButton = tkinter.Button(root, text='D', width=1, command=lambda: on_letter('D'), relief="flat", bg="white", font="tahoma")
    eButton = tkinter.Button(root, text='E', width=1, command=lambda: on_letter('E'), relief="flat", bg="white", font="tahoma")
    fButton = tkinter.Button(root, text='F', width=1, command=lambda: on_letter('F'), relief="flat", bg="white", font="tahoma")
    gButton = tkinter.Button(root, text='G', width=1, command=lambda: on_letter('G'), relief="flat", bg="white", font="tahoma")
    hButton = tkinter.Button(root, text='H', width=1, command=lambda: on_letter('H'), relief="flat", bg="white", font="tahoma")
    iButton = tkinter.Button(root, text='I', width=1, command=lambda: on_letter('I'), relief="flat", bg="white", font="tahoma")
    jButton = tkinter.Button(root, text='J', width=1, command=lambda: on_letter('J'), relief="flat", bg="white", font="tahoma")
    kButton = tkinter.Button(root, text='K', width=1, command=lambda: on_letter('K'), relief="flat", bg="white", font="tahoma")
    lButton = tkinter.Button(root, text='L', width=1, command=lambda: on_letter('L'), relief="flat", bg="white", font="tahoma")
    mButton = tkinter.Button(root, text='M', width=1, command=lambda: on_letter('M'), relief="flat", bg="white", font="tahoma")
    nButton = tkinter.Button(root, text='N', width=1, command=lambda: on_letter('N'), relief="flat", bg="white", font="tahoma")
    oButton = tkinter.Button(root, text='O', width=1, command=lambda: on_letter('O'), relief="flat", bg="white", font="tahoma")
    pButton = tkinter.Button(root, text='P', width=1, command=lambda: on_letter('P'), relief="flat", bg="white", font="tahoma")
    qButton = tkinter.Button(root, text='Q', width=1, command=lambda: on_letter('Q'), relief="flat", bg="white", font="tahoma")
    rButton = tkinter.Button(root, text='R', width=1, command=lambda: on_letter('R'), relief="flat", bg="white", font="tahoma")
    sButton = tkinter.Button(root, text='S', width=1, command=lambda: on_letter('S'), relief="flat", bg="white", font="tahoma")
    tButton = tkinter.Button(root, text='T', width=1, command=lambda: on_letter('T'), relief="flat", bg="white", font="tahoma")
    uButton = tkinter.Button(root, text='U', width=1, command=lambda: on_letter('U'), relief="flat", bg="white", font="tahoma")
    vButton = tkinter.Button(root, text='V', width=1, command=lambda: on_letter('V'), relief="flat", bg="white", font="tahoma")
    wButton = tkinter.Button(root, text='W', width=1, command=lambda: on_letter('W'), relief="flat", bg="white", font="tahoma")
    xButton = tkinter.Button(root, text='X', width=1, command=lambda: on_letter('X'), relief="flat", bg="white", font="tahoma")
    yButton = tkinter.Button(root, text='Y', width=1, command=lambda: on_letter('Y'), relief="flat", bg="white", font="tahoma")
    zButton = tkinter.Button(root, text='Z', width=1, command=lambda: on_letter('Z'), relief="flat", bg="white", font="tahoma")
    backspaceButton = tkinter.Button(root, text='⇐', width=7, command=on_backspace, relief="flat", bg="white", font="tahoma")
    faceNameText = tkinter.Text(root, height = 1, width = 60)
    
    thread = threading.Thread(target=stream, args=(mainCameraLabel,mainCamera,sideCameraLabel,sideCamera))
    thread.daemon = 1
    
    draw_main_screen()
    thread.start()
    root.mainloop()
