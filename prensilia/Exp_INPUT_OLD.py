from cv2 import namedWindow, imshow, waitKey, cvtColor, COLOR_RGB2BGR, imwrite
from hand_grabber import PyOpenNIHandGrabber
from pose_recognizer import PyPoseRecognizer
from my_fun import *
from sklearn.externals import joblib
from SerialClient import *
import time
from random import shuffle



WIDTH = 640
HEIGHT = 480
USE_CPU = False
USE_FORESTBIG = True
USE_PROB = True

SIGN_CNTR = 2


if __name__=="__main__":

    raspberryClient = ClientSocket('10.10.0.1', 9091, '/dev/ttyUSB0', 'P'*16)
        
    sign_set = {'s':[72,   5, 250, 250, 250, 250,   5], 'v':[72, 250,   5,   5, 250, 250,   5], 'u':[72, 250,   5,   5, 250, 250,   5], 'w':[72, 250,   5,   5,   5, 250,   5], 'rest':[ 72,  5,   5,   5,   5,   5,   5], 'f':[72, 200, 160,5,5,5,120]}

    signs = ['f', 'horns', 'rest', 's', 'u', 'v', 'w']
    
    
    all_sign_to_perform = [['f', 'v', 'f', 'v', 'v', 'f', 's', 'w', 'f', 'v', 'v', 'w', 'f', 'v', 's', 'v', 's', 'f', 's', 'w', 'w', 'w', 's', 's', 's', 'f', 'v', 'v', 's', 'w', 'f', 's', 'f', 'f', 'w', 's', 'w', 'w', 'w', 'v'], ['v', 'f', 'v', 'v', 'w', 'v', 'v', 's', 'v', 'f', 'w', 'v', 's', 'v', 'f', 's', 's', 'v', 's', 'v', 'w', 'w', 'f', 's', 'w', 'w', 'w', 's', 'f', 's', 'f', 'w', 's', 'f', 'w', 'f', 'f', 'f', 'w', 's'], ['f', 'v', 'f', 'f', 'w', 's', 'v', 's', 'v', 'v', 'w', 'v', 's', 's', 's', 'w', 'v', 's', 's', 'w', 'w', 'f', 's', 'f', 'w', 'v', 's', 'w', 'v', 'f', 'f', 'v', 'w', 'w', 'w', 's', 'v', 'f', 'f', 'f'], ['v', 'f', 'v', 'v', 's', 'w', 'w', 'v', 'w', 's', 's', 'w', 's', 's', 'v', 'f', 'v', 'v', 'w', 'f', 'v', 's', 's', 's', 'w', 'f', 'f', 'f', 'w', 'w', 'f', 'f', 'v', 'w', 'f', 's', 'w', 'f', 'v', 's'], ['s', 'w', 'w', 's', 'v', 'w', 's', 'f', 's', 'w', 's', 'v', 's', 's', 'v', 'w', 'f', 'w', 's', 'w', 'f', 'v', 'v', 'f', 'f', 'v', 'v', 's', 'f', 's', 'v', 'w', 'f', 'w', 'f', 'w', 'v', 'f', 'f', 'v'], ['f', 'v', 's', 'w', 'f', 'f', 'v', 's', 'v', 's', 'f', 'v', 'w', 'v', 's', 'v', 'v', 's', 'w', 'w', 'f', 'w', 'v', 'v', 's', 'w', 'f', 'f', 'f', 's', 'f', 'v', 's', 's', 'w', 'f', 'w', 's', 'w', 'w'], ['s', 'f', 'v', 'w', 'f', 's', 'v', 'v', 'v', 'f', 'w', 'f', 's', 'v', 'f', 'v', 's', 's', 'v', 's', 'f', 'f', 'f', 'w', 'w', 'w', 's', 'w', 's', 'w', 'w', 's', 'w', 'w', 'f', 'v', 'f', 'v', 's', 'v'], ['v', 'w', 'v', 'v', 'f', 'v', 's', 'w', 'v', 'f', 'v', 'f', 's', 's', 'w', 'v', 'f', 'w', 'w', 'f', 'v', 'w', 'f', 'w', 'w', 'f', 's', 'w', 's', 's', 'v', 's', 'f', 'f', 's', 'f', 's', 'w', 'v', 's'], ['f', 'w', 'w', 'f', 'f', 'v', 's', 'f', 'f', 'w', 'v', 's', 's', 's', 'w', 'v', 's', 'f', 's', 's', 'w', 's', 'f', 'w', 'v', 'v', 's', 'v', 'f', 's', 'w', 'w', 'v', 'f', 'w', 'f', 'w', 'v', 'v', 'v'], ['s', 'f', 's', 'v', 'f', 'w', 's', 'w', 'w', 'w', 'w', 'f', 'f', 'w', 's', 'w', 'f', 'v', 'v', 'w', 'v', 'f', 'v', 'v', 'w', 's', 'v', 'v', 's', 's', 'f', 'v', 'w', 's', 'v', 'f', 's', 's', 'f', 'f'], ['w', 'w', 'w', 'f', 'v', 'v', 'v', 'f', 's', 'w', 'f', 's', 'w', 's', 'v', 'v', 'v', 's', 'f', 'f', 'w', 'w', 'f', 'f', 's', 'v', 's', 'v', 's', 'w', 'f', 'v', 's', 's', 'f', 's', 'f', 'v', 'w', 'w'], ['s', 'w', 'v', 's', 's', 'w', 'f', 'v', 'v', 'f', 'f', 'f', 'v', 's', 'w', 'v', 'w', 's', 'f', 'f', 'w', 's', 'f', 's', 'f', 'w', 'w', 's', 'f', 'v', 's', 'v', 'w', 'f', 'v', 'v', 'w', 'v', 'w', 's'], ['v', 'w', 'f', 'v', 'w', 'v', 'f', 's', 'f', 'f', 'w', 'w', 's', 's', 's', 'v', 'v', 's', 's', 'v', 'w', 'f', 's', 'w', 'f', 'f', 's', 'f', 'v', 'f', 'f', 's', 's', 'w', 'w', 'v', 'v', 'v', 'w', 'w'], ['s', 's', 'f', 'v', 'f', 's', 'f', 'v', 'w', 'v', 'v', 'f', 'f', 'w', 's', 'f', 'v', 's', 'w', 'f', 'v', 's', 'w', 'w', 's', 'v', 's', 'w', 'f', 'f', 'v', 'v', 's', 'v', 's', 'w', 'w', 'w', 'w', 'f'], ['s', 'w', 'w', 'f', 'f', 'w', 'w', 's', 's', 'v', 's', 'w', 'w', 's', 's', 'f', 'v', 's', 'f', 'v', 'f', 'w', 'v', 'w', 'v', 'v', 'f', 'v', 's', 'v', 'v', 'w', 's', 'v', 'f', 's', 'f', 'f', 'w', 'f'], ['w', 'v', 'f', 'v', 'f', 's', 'v', 'f', 'v', 'f', 'v', 'w', 'v', 'w', 'v', 's', 'f', 'w', 'v', 'w', 's', 'v', 'v', 's', 'f', 's', 'w', 's', 'w', 'w', 'w', 'f', 's', 's', 's', 'f', 'f', 'f', 's', 'w'], ['f', 'w', 'v', 's', 'w', 'f', 'v', 'w', 'f', 'v', 'w', 'w', 's', 'v', 's', 's', 'w', 'v', 'v', 'f', 'f', 'w', 'f', 'f', 'f', 's', 's', 'w', 'w', 's', 'f', 's', 's', 'w', 'v', 's', 'v', 'f', 'v', 'v'], ['w', 'v', 'w', 'f', 'w', 's', 'v', 'f', 'v', 'f', 's', 's', 'v', 'v', 's', 'w', 's', 'f', 'w', 'v', 'f', 's', 'w', 's', 'f', 'w', 'v', 'w', 'f', 's', 's', 'w', 'v', 'v', 'v', 'w', 's', 'f', 'f', 'f'], ['v', 'w', 'w', 's', 's', 'v', 'f', 's', 'f', 's', 'v', 'v', 's', 'w', 'w', 'f', 'f', 'f', 'f', 'f', 's', 'v', 'v', 'v', 's', 'w', 'w', 'v', 'f', 'f', 's', 'w', 's', 'w', 'f', 'v', 'w', 'v', 'w', 's'], ['f', 'v', 'v', 'f', 'f', 'w', 'v', 'f', 's', 'w', 's', 'v', 'w', 'v', 'w', 'w', 'v', 'w', 'f', 's', 'w', 's', 'f', 's', 'w', 'w', 'f', 'v', 'f', 'v', 's', 'v', 's', 's', 'v', 'f', 's', 's', 'f', 'w'], ['w', 'v', 's', 'v', 's', 'v', 's', 's', 'v', 'v', 'w', 's', 'w', 'f', 'w', 'w', 's', 'f', 's', 'w', 's', 'v', 'f', 'f', 'f', 'f', 'w', 'v', 'w', 's', 'v', 'f', 'v', 'f', 'f', 'v', 'f', 'w', 'w', 's'], ['f', 'v', 's', 'f', 'v', 'f', 's', 'v', 'f', 'w', 's', 'v', 'v', 's', 'w', 'w', 'f', 's', 'w', 'f', 'v', 'w', 's', 's', 'f', 'w', 's', 'w', 'v', 'w', 'w', 'f', 's', 'w', 's', 'f', 'v', 'v', 'f', 'v'], ['s', 'v', 'f', 'f', 'v', 'v', 'f', 'w', 'f', 'w', 'v', 'v', 'f', 'w', 'f', 'v', 's', 's', 's', 'w', 's', 'v', 'v', 's', 'f', 'w', 'w', 'f', 'v', 's', 'f', 's', 'v', 'w', 'w', 's', 'w', 's', 'f', 'w'], ['s', 'w', 's', 'w', 'w', 's', 'v', 'w', 's', 'v', 'f', 'f', 'v', 's', 'w', 'w', 'f', 'f', 'f', 'w', 'w', 'v', 's', 'f', 'v', 'v', 'f', 'f', 'w', 'w', 'v', 's', 'v', 's', 's', 'v', 'f', 'v', 's', 'f'], ['w', 'v', 'v', 'f', 'w', 'w', 's', 's', 's', 'v', 'v', 'w', 'w', 's', 'f', 'w', 'v', 'v', 'w', 'v', 'v', 'f', 'f', 'f', 'f', 'f', 'v', 'w', 's', 's', 'w', 'f', 's', 's', 'f', 'w', 's', 'f', 's', 'v'], ['w', 'v', 'v', 'w', 'w', 'f', 'w', 's', 'v', 'f', 'w', 's', 's', 'v', 'v', 'w', 's', 's', 'w', 'w', 's', 'f', 'f', 's', 'v', 'v', 'w', 'w', 's', 'f', 'f', 'v', 'v', 's', 'f', 's', 'f', 'f', 'v', 'f'], ['v', 'w', 'w', 'v', 'v', 'f', 'w', 'f', 's', 's', 'f', 's', 's', 'w', 's', 's', 'w', 's', 'w', 'v', 'f', 'f', 'v', 'f', 'w', 'w', 'v', 'f', 's', 'f', 'w', 'v', 'f', 's', 'w', 'f', 'v', 's', 'v', 'v'], ['v', 'f', 'v', 'w', 'w', 'f', 'w', 'v', 'v', 'w', 's', 'w', 'v', 's', 'v', 'f', 'v', 'v', 'f', 's', 'w', 'f', 'w', 'w', 'f', 'w', 's', 's', 's', 'v', 's', 'v', 'w', 's', 's', 'f', 'f', 'f', 'f', 's'], ['v', 's', 's', 'f', 'f', 'v', 'w', 'f', 'f', 'f', 'v', 'v', 'f', 'v', 'w', 'w', 'f', 'f', 'w', 's', 'v', 'w', 's', 'f', 's', 'w', 's', 'f', 'v', 's', 'w', 'w', 'w', 'v', 's', 'v', 's', 'w', 'v', 's'], ['s', 'v', 'f', 's', 'w', 's', 'w', 'v', 'w', 'v', 'v', 'f', 's', 'v', 'f', 'v', 'w', 'w', 'w', 'v', 'f', 's', 'f', 'f', 'f', 'v', 'w', 's', 'v', 'w', 's', 's', 'w', 'f', 'f', 's', 'v', 'f', 'w', 's'], ['w', 'f', 'f', 's', 'w', 's', 's', 'v', 'w', 'v', 'w', 's', 'f', 'v', 'f', 'w', 's', 's', 'w', 'f', 'f', 'w', 'w', 's', 'v', 'v', 'v', 's', 'w', 's', 'f', 'f', 'v', 'f', 'v', 'v', 's', 'w', 'v', 'f'], ['s', 'f', 'f', 's', 's', 's', 'w', 'v', 'w', 'f', 'w', 'v', 's', 'f', 'f', 'f', 'v', 'v', 'f', 'v', 's', 'v', 'w', 's', 'w', 'w', 's', 'f', 'v', 'w', 'w', 'v', 's', 's', 'v', 'w', 'w', 'f', 'f', 'v'], ['w', 'w', 'w', 'w', 'f', 'f', 'v', 'w', 's', 'f', 'f', 'w', 'w', 's', 'v', 's', 'v', 'v', 's', 's', 'w', 's', 'w', 'f', 's', 'f', 'f', 's', 'f', 'v', 'v', 's', 'w', 'v', 'v', 'f', 'v', 'v', 's', 'f'], ['w', 'v', 'w', 'w', 'v', 'w', 'v', 'f', 'f', 'f', 'f', 'f', 'v', 's', 'f', 'v', 'w', 'v', 'w', 'w', 'v', 'f', 's', 'f', 's', 'f', 'v', 's', 'v', 'w', 'f', 's', 'w', 's', 's', 's', 'w', 's', 's', 'v'], ['f', 's', 's', 's', 'v', 's', 'f', 'v', 's', 'w', 'w', 'f', 'f', 's', 'v', 'w', 's', 'w', 's', 'v', 'f', 'w', 'w', 'v', 'f', 'f', 's', 'w', 'v', 'v', 'v', 'f', 'f', 'w', 'f', 'w', 's', 'w', 'v', 'v'], ['w', 'w', 's', 'w', 'w', 'f', 'v', 's', 'f', 'v', 'f', 'w', 's', 'w', 's', 's', 'f', 'f', 'w', 's', 'v', 'w', 'w', 'f', 's', 'f', 'v', 'v', 'f', 'v', 'w', 's', 'f', 's', 'v', 'v', 'v', 's', 'f', 'v'], ['w', 'f', 'w', 's', 'v', 'v', 'f', 'w', 'v', 'f', 'w', 's', 's', 'f', 'v', 'v', 'f', 'f', 'f', 'f', 'w', 'v', 'f', 'w', 'w', 'w', 'v', 's', 's', 'f', 's', 's', 's', 'v', 'w', 'v', 's', 'v', 'w', 's'], ['v', 'v', 'v', 'w', 'w', 'w', 'f', 'f', 'f', 's', 'w', 'v', 's', 'w', 'w', 'f', 's', 's', 'w', 'v', 'f', 's', 'v', 'v', 'w', 's', 'v', 'w', 'v', 'f', 'v', 'w', 'f', 'f', 's', 'f', 's', 's', 's', 'f'], ['v', 'w', 'w', 'w', 'w', 'w', 'f', 'w', 's', 'v', 'v', 'f', 'v', 'v', 'f', 'f', 's', 'w', 'f', 'f', 'v', 'v', 'w', 's', 's', 's', 'v', 'f', 's', 'v', 's', 'w', 'f', 's', 'f', 'w', 's', 'f', 's', 'v'], ['v', 'w', 'v', 'w', 'w', 'v', 'w', 'w', 'v', 'v', 'f', 'v', 's', 's', 's', 'f', 'v', 'v', 's', 'v', 'f', 's', 's', 'w', 'f', 's', 'f', 'f', 'f', 'w', 's', 's', 's', 'w', 'v', 'f', 'f', 'w', 'w', 'f'], ['v', 'w', 's', 's', 'f', 's', 's', 'v', 'v', 'f', 'w', 'v', 'w', 'w', 'f', 'v', 's', 'f', 's', 'f', 'v', 's', 's', 'f', 'w', 'w', 'f', 'w', 'f', 'v', 'f', 'w', 's', 'w', 'w', 'v', 'f', 'v', 's', 'v'], ['w', 's', 'v', 'w', 'f', 'w', 'v', 'v', 'v', 'v', 'w', 's', 'f', 'f', 'v', 'v', 'w', 'w', 'f', 'f', 's', 's', 's', 'f', 's', 'v', 'w', 'f', 's', 'v', 'w', 's', 'w', 'w', 's', 'f', 'f', 'f', 'v', 's'], ['v', 's', 'v', 'v', 's', 'v', 's', 'w', 'f', 'f', 'w', 'v', 'f', 'w', 'w', 'v', 's', 'f', 's', 'w', 'f', 'w', 'f', 's', 's', 'f', 'v', 's', 'v', 's', 's', 'f', 'w', 'w', 'w', 'f', 'w', 'v', 'f', 'v'], ['s', 's', 'v', 'f', 'w', 's', 's', 'w', 'v', 'v', 'w', 'v', 'w', 'v', 'v', 'w', 's', 's', 'v', 'f', 's', 'v', 'f', 'v', 's', 's', 'w', 'f', 'f', 'w', 'w', 's', 'f', 'f', 'v', 'w', 'w', 'f', 'f', 'f'], ['s', 'w', 's', 'v', 'v', 'f', 'f', 'v', 'f', 's', 'f', 's', 's', 'v', 'f', 'v', 'f', 'v', 'w', 's', 'f', 'w', 'v', 'f', 's', 'f', 'v', 'w', 'w', 'w', 'w', 's', 'f', 'v', 'w', 'w', 's', 'v', 's', 'w'], ['w', 'f', 's', 'v', 'v', 'f', 'f', 'f', 'w', 's', 'w', 'w', 'f', 'v', 'v', 'v', 's', 'w', 'v', 'f', 's', 's', 'v', 'f', 'v', 'f', 's', 's', 'v', 'w', 's', 'f', 'w', 'w', 'f', 's', 'v', 'w', 's', 'w'], ['w', 's', 'f', 's', 's', 'w', 's', 'v', 'w', 's', 'w', 'v', 'f', 'f', 's', 'v', 'v', 'f', 'f', 'v', 'f', 's', 'f', 'v', 'w', 'f', 'w', 's', 'w', 'w', 'w', 'v', 's', 'f', 'v', 'f', 's', 'w', 'v', 'v'], ['v', 'v', 'w', 's', 's', 'f', 'v', 'w', 'f', 'w', 's', 'f', 's', 'w', 'w', 's', 'w', 'w', 'v', 'f', 'v', 'f', 's', 'f', 'w', 'v', 'v', 'f', 's', 's', 'w', 'v', 'v', 'w', 'v', 'f', 'f', 'f', 's', 's'], ['s', 'v', 'v', 's', 'w', 'w', 'v', 'f', 'v', 'v', 's', 'w', 'v', 'f', 's', 'f', 'f', 's', 'w', 's', 'f', 'w', 's', 'f', 'v', 'f', 'v', 'w', 's', 'w', 'w', 'f', 'v', 's', 'w', 'f', 'f', 'v', 'w', 's'], ['v', 's', 'f', 'f', 'w', 'v', 'w', 'w', 's', 's', 'v', 'v', 'v', 'f', 'w', 'v', 'v', 's', 'w', 's', 'f', 'f', 'f', 'w', 'w', 'v', 'w', 's', 'f', 's', 's', 'w', 'f', 'f', 'v', 'w', 's', 'f', 's', 'v'], ['w', 's', 'f', 'w', 'v', 's', 's', 'f', 'v', 'f', 'w', 'v', 'f', 'f', 'v', 'v', 's', 'v', 'w', 'v', 'f', 'v', 'f', 'f', 'w', 'w', 's', 's', 's', 'w', 'v', 'w', 's', 'w', 's', 'w', 's', 'f', 'v', 'f'], ['w', 'v', 'f', 'f', 'f', 'w', 'v', 'v', 'f', 's', 'f', 'w', 'w', 's', 's', 'w', 'f', 'v', 'v', 'w', 's', 'v', 'v', 'w', 's', 'v', 's', 'w', 'w', 'f', 's', 'w', 'v', 's', 'f', 'f', 's', 's', 'f', 'v'], ['w', 'f', 'w', 's', 'f', 's', 'v', 's', 'f', 'w', 's', 'f', 'w', 'w', 'w', 'v', 'v', 'v', 'f', 'v', 's', 'v', 'w', 's', 'w', 'v', 'f', 'f', 'v', 's', 'w', 's', 'f', 'v', 'f', 's', 'v', 'f', 's', 'w'], ['w', 's', 'f', 'w', 'v', 'v', 'w', 'f', 'f', 's', 'f', 'v', 'v', 's', 'w', 's', 'v', 'f', 'v', 'f', 'w', 'w', 's', 'w', 'f', 'v', 's', 's', 'f', 'w', 'f', 'v', 'v', 'w', 'f', 's', 'v', 's', 'w', 's'], ['w', 's', 's', 'w', 'f', 's', 'f', 'v', 'f', 'v', 'f', 'w', 'v', 'v', 'w', 's', 'w', 's', 'v', 'f', 's', 'v', 'w', 'w', 'f', 'f', 'w', 'v', 's', 'f', 'v', 'w', 'v', 's', 's', 'v', 'f', 'f', 'w', 's'], ['v', 'f', 'v', 'w', 'v', 'v', 's', 'f', 'w', 'w', 'f', 's', 'f', 's', 'v', 's', 's', 'v', 'w', 's', 'w', 'v', 'f', 'w', 'w', 'v', 'w', 's', 's', 'f', 'f', 's', 'f', 'v', 's', 'f', 'f', 'v', 'w', 'w'], ['w', 'w', 'w', 's', 'w', 's', 'v', 'w', 'v', 'f', 'f', 'f', 'v', 'v', 'w', 'f', 'f', 'v', 's', 'v', 'v', 's', 'w', 's', 'f', 'f', 'f', 'v', 'v', 'v', 's', 'w', 'w', 's', 's', 'w', 'f', 's', 'f', 's'], ['w', 's', 's', 's', 'w', 's', 'w', 'f', 'f', 'w', 'f', 'w', 'w', 'f', 'v', 'v', 'w', 'f', 's', 'f', 's', 'f', 'w', 'w', 'v', 'f', 'w', 's', 'v', 'v', 'v', 's', 'f', 'v', 's', 'f', 'v', 'v', 'v', 's'], ['v', 's', 's', 'v', 'f', 's', 'v', 'w', 's', 'v', 'f', 'w', 'f', 'v', 'w', 'w', 's', 'w', 'w', 's', 'f', 'w', 'f', 's', 'f', 'f', 'w', 'f', 'w', 's', 's', 'f', 's', 'v', 'v', 'f', 'w', 'v', 'v', 'v'], ['s', 'v', 'w', 's', 'f', 'f', 's', 'w', 's', 'w', 'v', 'v', 'w', 'w', 'v', 'f', 's', 'v', 'f', 'f', 'f', 'v', 'v', 'f', 'v', 'v', 'w', 'w', 'w', 's', 'f', 's', 's', 's', 'f', 'w', 'f', 's', 'v', 'w'], ['s', 's', 'v', 'w', 'f', 'w', 'f', 'w', 'v', 'v', 'f', 'w', 'w', 'f', 'f', 'v', 'f', 's', 'w', 'f', 'v', 's', 'v', 'v', 'v', 's', 'f', 'w', 'f', 's', 's', 's', 's', 'w', 'f', 's', 'v', 'v', 'w', 'w'], ['w', 's', 'v', 'w', 'w', 'w', 'v', 's', 'w', 'v', 'v', 's', 's', 'v', 's', 's', 'v', 's', 'w', 'f', 'f', 'w', 's', 'v', 'f', 'f', 'v', 'v', 'f', 's', 'f', 'f', 'v', 'w', 'f', 'w', 'f', 's', 'f', 'w'], ['v', 'w', 'w', 'f', 'v', 'v', 's', 's', 's', 'v', 's', 'f', 'f', 's', 's', 'f', 'w', 's', 'f', 'v', 's', 'f', 'f', 'f', 'w', 'w', 'w', 's', 'v', 'w', 'w', 'w', 'v', 's', 'v', 'f', 'v', 'v', 'w', 'f'], ['w', 'f', 'v', 'f', 'w', 'v', 's', 'f', 'v', 'f', 'f', 'v', 'f', 's', 's', 'f', 'v', 's', 's', 'w', 'v', 'f', 'v', 'w', 's', 'w', 'w', 'v', 's', 'f', 's', 'f', 's', 'w', 's', 'v', 'w', 'w', 'w', 'v'], ['w', 'v', 'f', 'f', 'v', 's', 's', 's', 'w', 'w', 's', 'w', 'v', 'w', 'f', 'f', 'f', 's', 'f', 's', 'v', 'w', 's', 'w', 'f', 'f', 's', 'v', 'v', 'w', 'v', 'w', 's', 'v', 's', 'v', 'f', 'f', 'w', 'v'], ['s', 's', 'w', 'w', 'w', 'w', 'f', 'f', 'v', 's', 's', 's', 'f', 'f', 'v', 'f', 'v', 'v', 's', 'f', 'w', 's', 'w', 'v', 'v', 's', 'f', 'v', 'f', 'w', 'f', 'v', 'w', 'w', 's', 'v', 's', 'w', 'v', 'f'], ['f', 'f', 's', 's', 's', 'v', 'w', 's', 's', 'w', 'f', 'f', 'f', 'w', 's', 'w', 'w', 's', 's', 's', 'f', 'v', 'f', 'w', 'v', 'v', 'w', 'v', 'v', 'w', 'v', 'v', 'f', 'v', 's', 'f', 'w', 'v', 'w', 'f'], ['w', 'v', 's', 'f', 's', 's', 'v', 'f', 's', 'v', 's', 's', 'f', 'v', 'w', 'f', 'f', 'v', 'w', 'f', 'v', 'w', 'f', 's', 'f', 'f', 'w', 's', 'f', 'v', 's', 'w', 'v', 'w', 'w', 'v', 'w', 'v', 'w', 's'], ['s', 'f', 'w', 's', 'v', 'w', 's', 'f', 'w', 'w', 's', 'v', 's', 'v', 'w', 's', 'v', 's', 'w', 'f', 'w', 'w', 'f', 'v', 'f', 'v', 's', 'w', 'f', 'v', 's', 'v', 'f', 's', 'v', 'f', 'v', 'f', 'f', 'w'], ['s', 'w', 'f', 'v', 's', 'v', 's', 's', 'f', 'f', 'v', 's', 'v', 'f', 'w', 'v', 's', 'w', 'v', 'f', 'f', 'f', 'v', 'w', 'w', 'w', 'v', 'f', 'w', 'v', 'v', 's', 'w', 'f', 's', 's', 'f', 'w', 's', 'w'], ['f', 'w', 's', 'f', 'w', 'f', 'v', 'f', 'w', 'v', 'w', 's', 's', 's', 'w', 'v', 'v', 'v', 's', 'v', 'f', 's', 'f', 'v', 'f', 'w', 'v', 'f', 'w', 'w', 's', 'f', 's', 'w', 's', 'v', 'w', 'v', 's', 'f'], ['s', 'f', 'w', 'v', 'w', 's', 's', 's', 's', 'w', 'v', 'v', 'f', 'w', 'w', 'f', 'f', 'v', 'f', 'f', 'v', 'f', 's', 'v', 'w', 'f', 'v', 's', 'w', 'v', 's', 'v', 'f', 'f', 'w', 's', 'v', 'w', 's', 'w'], ['w', 's', 's', 'f', 'v', 'v', 'w', 'v', 'f', 'w', 's', 's', 'f', 's', 's', 'w', 'f', 'v', 'f', 'w', 'v', 'f', 'v', 'f', 'v', 's', 'w', 's', 'w', 'w', 'w', 'v', 'v', 's', 'f', 'v', 'w', 'f', 's', 'f'], ['f', 'v', 's', 'w', 's', 'f', 'w', 'f', 'f', 'f', 'w', 'w', 'v', 'v', 'v', 's', 'f', 'v', 'w', 's', 'w', 'v', 'v', 'w', 's', 'w', 's', 'w', 's', 's', 'w', 's', 'f', 'f', 'f', 'v', 'v', 'f', 's', 'v'], ['f', 'v', 'f', 's', 'w', 'v', 'w', 's', 'f', 's', 's', 'w', 'f', 'v', 'v', 'f', 'v', 'w', 'f', 'w', 's', 's', 'w', 'f', 'v', 'w', 'v', 's', 's', 'v', 'v', 'f', 'w', 's', 'w', 'w', 's', 'v', 'f', 'f'], ['w', 'f', 'w', 'f', 'f', 'v', 's', 'w', 'v', 'f', 's', 'v', 's', 'v', 's', 's', 'f', 'w', 's', 'w', 'w', 'v', 'w', 'v', 'w', 's', 'v', 'f', 'f', 'f', 's', 'w', 'v', 'f', 's', 'f', 'w', 'v', 's', 'v'], ['v', 's', 's', 'w', 'w', 's', 's', 'v', 'f', 'f', 'w', 's', 'f', 'w', 'f', 'w', 'f', 'w', 'v', 'f', 's', 'v', 's', 's', 's', 'v', 's', 'f', 'w', 'f', 'v', 'v', 'w', 'w', 'v', 'v', 'v', 'f', 'f', 'w'], ['w', 'v', 'f', 'v', 's', 'v', 'v', 's', 'v', 'v', 'w', 's', 'w', 'v', 'f', 'f', 'f', 's', 'v', 'w', 's', 's', 'w', 'w', 's', 's', 'v', 'w', 'w', 'f', 'w', 'w', 'f', 'f', 'f', 's', 'f', 'f', 's', 'v'], ['v', 'w', 'w', 'w', 'w', 'v', 's', 'f', 'f', 'w', 's', 's', 's', 'v', 'v', 'w', 's', 's', 'f', 'w', 'f', 'v', 'v', 'f', 'f', 'v', 's', 's', 'f', 'f', 'w', 'w', 'f', 'v', 'v', 'w', 's', 's', 'v', 'f'], ['v', 'v', 'v', 'w', 'w', 'f', 'w', 's', 'w', 'v', 'f', 'w', 'f', 'f', 's', 's', 's', 'w', 's', 'v', 'w', 'w', 'v', 'v', 'v', 'f', 's', 'f', 's', 'f', 's', 's', 'v', 'f', 'w', 'f', 's', 'w', 'v', 'f'], ['s', 'f', 'v', 'f', 'v', 's', 'v', 's', 'f', 'w', 's', 'w', 'w', 'f', 's', 's', 'v', 'f', 'w', 'w', 'w', 'w', 'w', 'v', 'w', 'f', 'f', 'v', 'v', 'f', 's', 'v', 'f', 'w', 'f', 'v', 'v', 's', 's', 's'], ['f', 'f', 's', 'v', 'v', 'f', 'w', 'w', 's', 's', 's', 'f', 's', 'v', 's', 'w', 'f', 'f', 'w', 'v', 's', 'f', 'w', 'w', 'w', 'f', 'v', 'v', 's', 's', 'w', 's', 'w', 'v', 'f', 'f', 'w', 'v', 'v', 'v'], ['w', 'f', 'f', 's', 'w', 's', 'w', 's', 'v', 'w', 's', 'v', 'f', 'f', 'v', 'f', 'v', 'v', 's', 'w', 'f', 'f', 'w', 's', 'w', 'w', 's', 'f', 's', 's', 'v', 'w', 'v', 'f', 'v', 'f', 'w', 'v', 's', 'v'], ['f', 's', 'w', 'f', 'f', 'f', 's', 'w', 's', 'f', 'w', 'v', 's', 's', 'f', 's', 's', 'v', 'v', 'f', 's', 's', 'w', 'f', 'w', 'v', 'w', 's', 'v', 'f', 'f', 'v', 'v', 'v', 'v', 'w', 'w', 'w', 'v', 'w'], ['w', 'w', 's', 's', 'f', 's', 's', 'f', 'f', 'v', 'w', 'w', 's', 'f', 'w', 'v', 'v', 'f', 'w', 'w', 'v', 'v', 's', 'f', 's', 'v', 'v', 's', 'v', 's', 'w', 'w', 'v', 'v', 's', 'f', 'f', 'f', 'w', 'f'], ['w', 'f', 'f', 's', 's', 'f', 'w', 'v', 'v', 'f', 'w', 's', 'w', 'w', 'v', 'v', 'f', 'v', 'w', 'w', 'v', 'f', 'f', 'w', 'w', 's', 'v', 's', 'v', 'w', 'f', 's', 'v', 'f', 'v', 'f', 's', 's', 's', 's'], ['w', 'f', 'w', 'w', 's', 'f', 'v', 's', 'f', 'f', 's', 's', 'w', 'v', 's', 'w', 'f', 'v', 'f', 's', 'w', 'w', 'w', 's', 'v', 'v', 'v', 'v', 'f', 'w', 's', 's', 'f', 'v', 'f', 'v', 'v', 's', 'w', 'f'], ['v', 'f', 'f', 'f', 'f', 'w', 'v', 'v', 's', 's', 'w', 'f', 'f', 'w', 's', 'w', 's', 'v', 'w', 'w', 's', 'f', 's', 's', 'v', 'v', 'v', 'v', 'w', 's', 's', 'f', 'v', 'v', 'w', 'f', 'w', 'w', 's', 'f'], ['f', 'f', 'w', 'v', 'w', 'v', 's', 's', 'v', 'w', 'f', 'f', 's', 's', 'f', 'v', 'w', 'w', 'v', 'v', 's', 'w', 'f', 'f', 'w', 's', 'f', 'v', 's', 'v', 'v', 'f', 's', 'v', 'w', 's', 'w', 's', 'w', 'f'], ['f', 'v', 'v', 'w', 's', 's', 's', 's', 'f', 's', 'w', 'v', 'w', 'v', 'f', 'v', 'w', 's', 'w', 'w', 'w', 'f', 's', 'f', 'v', 'w', 'v', 's', 'w', 'v', 'f', 'v', 'f', 'f', 's', 's', 'f', 'w', 'f', 'v'], ['s', 'w', 'v', 's', 'v', 'f', 'f', 'w', 'w', 's', 'w', 'w', 'f', 's', 's', 'v', 'f', 'w', 's', 'v', 'v', 'w', 's', 'w', 'f', 'v', 'f', 'v', 's', 'f', 'f', 'v', 'w', 'f', 's', 'w', 'f', 'v', 's', 'v'], ['f', 'f', 'v', 's', 's', 'w', 'w', 'w', 'f', 'w', 'v', 'v', 's', 'v', 'v', 's', 'w', 'v', 'v', 'f', 's', 's', 'v', 'f', 'f', 's', 'w', 'w', 'f', 'v', 'f', 'f', 's', 'v', 'f', 'w', 'w', 'w', 's', 's'], ['v', 'w', 'w', 'f', 'v', 's', 'f', 'v', 's', 'v', 's', 'w', 's', 'f', 's', 'f', 'w', 'f', 's', 'f', 's', 'w', 'v', 'w', 'w', 's', 'w', 's', 'w', 's', 'v', 'w', 'f', 'f', 'v', 'f', 'v', 'v', 'v', 'f'], ['f', 'w', 's', 's', 'w', 's', 's', 'v', 'f', 'f', 's', 'w', 'v', 'v', 'f', 'v', 'v', 'v', 'w', 'f', 'w', 'v', 'f', 'w', 'w', 'f', 'v', 'w', 's', 's', 'v', 's', 'f', 'v', 'w', 's', 'f', 's', 'f', 'w'], ['f', 'f', 's', 'f', 'w', 'f', 'v', 's', 'v', 'f', 'v', 'w', 'w', 'v', 'w', 's', 'w', 's', 'v', 'w', 'f', 'f', 'w', 'f', 'f', 'v', 's', 'w', 's', 'v', 's', 'v', 'v', 's', 'w', 'f', 's', 'v', 's', 'w'], ['s', 'f', 'w', 'f', 'v', 'f', 's', 'w', 'v', 'w', 'w', 'v', 's', 'f', 'w', 'f', 'w', 'f', 'w', 's', 'f', 'f', 'v', 'f', 'v', 's', 's', 'v', 'w', 'w', 'v', 'v', 's', 'w', 'v', 's', 'f', 'v', 's', 's'], ['v', 'f', 'v', 'f', 'f', 's', 'w', 'w', 'w', 'f', 'f', 'w', 'f', 's', 'w', 'f', 's', 'w', 'v', 's', 's', 'v', 's', 'w', 'f', 'v', 'f', 's', 'v', 's', 'w', 'f', 'v', 'w', 'w', 'v', 's', 'v', 'v', 's'], ['v', 'w', 'w', 'v', 'v', 'v', 'f', 's', 'w', 's', 'f', 'v', 'v', 'w', 'w', 'w', 'w', 'v', 'w', 'f', 's', 's', 's', 'f', 'f', 's', 'f', 'f', 's', 'w', 'v', 's', 'f', 'w', 'f', 'v', 'v', 'f', 's', 's'], ['v', 'v', 'f', 'f', 's', 's', 'f', 'w', 'v', 'v', 's', 'f', 'v', 'v', 'w', 'f', 'f', 'w', 'w', 'v', 's', 'f', 's', 'v', 's', 'w', 's', 'w', 'w', 'f', 'w', 'v', 'f', 'w', 's', 's', 'w', 'v', 'f', 's'], ['f', 'f', 's', 's', 's', 'f', 'v', 'w', 's', 's', 'w', 'f', 'w', 'w', 's', 'v', 'w', 'w', 'v', 'v', 'v', 's', 'f', 'v', 'f', 'v', 'f', 'v', 's', 'f', 'v', 'f', 'w', 'w', 'v', 'f', 'w', 's', 's', 'w']]       
    
    sign_to_perform = all_sign_to_perform[SIGN_CNTR]

    namedWindow("rgb")
    namedWindow("prediction")

    grabber = PyOpenNIHandGrabber()
    streamWidth = WIDTH
    streamHeight = HEIGHT
    trainingSetSamplesWidth = 320
    radius = 150

    if (USE_FORESTBIG):
        recog = PyPoseRecognizer(streamWidth, streamHeight, "../forest-final.xml", USE_CPU, trainingSetSamplesWidth)
    else:
        recog = PyPoseRecognizer(streamWidth, streamHeight, "./forest.xml", USE_CPU, trainingSetSamplesWidth)
                             
    clf = joblib.load('Random_class_95.pkl')
    
    f = open('ParlomaSender.dat', 'a')
    f.write('Start experiment, seq number ='+str(SIGN_CNTR) +'\n')
    f.write('Sign to perform: ' + " - ".join(sign_to_perform) + '\n')

    counter = 0
    while True:
        if (counter >= len(sign_to_perform)):
            print 'Experiment Terminated'
            f.write('Stop experiment\n')
            f.close()
            break
        print("Wave the hand in front of the sensor")
        while True:
            rgb, depth = grabber.grabFrames()
            pos = grabber.getHand3DPos()     
            if pos[0] or pos[1] or pos[2]:
                break

        
        print 'Hand Recognized'
        print 'Please perform sign ', sign_to_perform[counter]
        supposed_sign = sign_to_perform[counter]
        counter += 1
        firstPose = True

        actual_sign = 'rest'
        new_sign = 'rest'
        sign_counter = 0

        while True:
            rgb, depth = grabber.grabFrames()

            pos = grabber.getHand3DPos()
            mask = grabber.segment(depth, pos, radius)
            prediction = recog.predict(depth, mask)
            joints = recog.getJoints(depth, mask)
            
            imwrite("rgb" + str(time.time()) + ".jpg", cvtColor(rgb, COLOR_RGB2BGR))
            imwrite("prediction"+ str(time.time()) + ".jpg", prediction)

            
            if (USE_PROB):
                prob = clf.predict_proba([joints2dist(joints)])
                mm = prob.argmax()
                sign = signs[mm]
                prob = prob[0][mm]
            else:
                sign = clf.predict([joints2dist(joints)])
                sign = sign.tolist()[0]
                prob = 1
            
            if(prob <= 0.3):
#                print "No sign recognized"
                sign_counter = 0
               # actual_sign = 'rest'
            else: 
                #print sign
                if (sign != actual_sign):
                    if (sign == new_sign):
                        sign_counter += 1
                        if (sign_counter == 5):
                            if (sign in sign_set):
                                msg = ''.join([chr(i) for i in sign_set[sign]])
                                raspberryClient.send_msg(msg,0)
                                actual_sign = sign
                                f.write(str(time.time()) + '\t' + actual_sign + '\t'+ supposed_sign + '\n' )
                                print 'Sent Sign: ', actual_sign;
                                time.sleep(5)
                                msg = ''.join([chr(i) for i in sign_set['rest']])
                                raspberryClient.send_msg(msg,0)
                                f.write(str(time.time()) + '\t' + 'reset' + '\t'+ 'reset' +  '\n' )
                                if (counter < len(sign_to_perform)):
                                    print 'Ready to perform sign ', sign_to_perform[counter], '\n'
                                    time.sleep(2)
                                break
                    else:
                        sign_counter = 0
                        new_sign = sign
                        

            imshow("rgb", cvtColor(rgb, COLOR_RGB2BGR))
            imshow("prediction", prediction)
            



            waitKey(1)

            if not pos[0] and not pos[1] and not pos[2]:
                print ("Hand position lost. Wave hand to restart ...")
                raspberryClient.send_msg(''.join([chr(i) for i in sign_set['rest']]),0)
                break
