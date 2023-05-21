import sys
import os
sys.path.append(f'{os.getcwd()}/src')
from raspberry_stereo.controllers.MainController import Controller

if __name__ == '__main__':
    controller = Controller()
    controller.run()
    