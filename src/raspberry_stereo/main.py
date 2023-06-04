import argparse
import sys
import os
sys.path.append(f'{os.getcwd()}/src')
from raspberry_stereo.controllers.MainController import MainController
import logging

def setup_command_line_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Run app in debug mode"
    )
    parser.add_argument(
        "--train-from-source",
        action="store_true",
        dest="train_from_source",
        help="Train and test system from dataset"
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = setup_command_line_parser()
    if args.debug:
        logging.basicConfig(level = logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
    else:
        logging.basicConfig(level = logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    controller = MainController(args)
    controller.run()
    