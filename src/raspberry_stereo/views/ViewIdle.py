import tkinter

class ViewIdle():
    def __init__(self,root):
        self.alreadyCheckedInLabel = tkinter.Label(root, text="", borderwidth=0, bg="black", fg="white")
        self.checkedInList = tkinter.Label(root, text="", borderwidth=0, bg="black", fg="white", justify=tkinter.LEFT)
        self.mainCameraLabel = tkinter.Label(root, borderwidth=0)
        self.sideCameraLabel = tkinter.Label(root, borderwidth=0)
        self.registerNewFaceButton = tkinter.Button(root, text='Register', width=8, relief="flat", bg="white", font="tahoma")
        self.welcomeMessageLabel = tkinter.Label(root, text="", borderwidth=0, height=12, width=38, background='green',font=("tahoma",25), fg="white")
