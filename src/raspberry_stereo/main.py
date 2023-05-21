import sys
import os
sys.path.append(f'{os.getcwd()}/src')
from raspberry_stereo.controllers.MainController import MainController

if __name__ == '__main__':
    controller = MainController()
    controller.run()
    