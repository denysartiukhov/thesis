import tkinter

class ViewIdle():
    def __init__(self,root):
        self.checked_in_list_header_label = tkinter.Label(root, text="Zalogowani użytkownicy:", borderwidth=0, bg="black", fg="white")
        self.already_checked_in_label = tkinter.Label(root, text="", borderwidth=0, bg="black", fg="white")
        self.checked_in_list_label = tkinter.Label(root, text="", borderwidth=0, bg="black", fg="white", justify=tkinter.LEFT)
        self.main_camera_label = tkinter.Label(root, borderwidth=0)
        self.side_camera_label = tkinter.Label(root, borderwidth=0)
        self.register_new_face_button = tkinter.Button(root, text='Rejestracja', width=8, relief="flat", bg="white", font="tahoma")
        self.welcome_message_label = tkinter.Label(root, text="", borderwidth=0, height=12, width=38, background='green',font=("tahoma",25), fg="white")
