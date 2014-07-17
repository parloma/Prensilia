from cv2 import namedWindow, imshow, waitKey, cvtColor, COLOR_RGB2BGR
from hand_grabber import PyOpenNIHandGrabber
from pose_recognizer import PyPoseRecognizer
from my_fun import *
from sklearn.externals import joblib
from hand import *
import time
import socket








WIDTH = 640
HEIGHT = 480
USE_CPU = False


#clientsocket=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#clientsocket.connect(('10.10.0.1',8089))



if __name__=="__main__":

    namedWindow("rgb")
    namedWindow("prediction")

    grabber = PyOpenNIHandGrabber()
    streamWidth = WIDTH
    streamHeight = HEIGHT
    trainingSetSamplesWidth = 320
    radius = 150

    #hand = start_communication()
    #open_hand(hand)

    recog = PyPoseRecognizer(streamWidth, streamHeight,
                             "../forest-final.xml",
                             USE_CPU, trainingSetSamplesWidth)
                             
    clf = joblib.load('tree_Random_class_4.pkl')

    print("Wave the hand in front of the sensor")

    while True:
        rgb, depth = grabber.grabFrames()
        pos = grabber.getHand3DPos()     
        if pos[0] or pos[1] or pos[2]:
            break



    print("CTRL+C to exit")
    firstPose = True

    actual_sign = 'rest'

    while True:
        rgb, depth = grabber.grabFrames()

        pos = grabber.getHand3DPos()
        mask = grabber.segment(depth, pos, radius)
        prediction = recog.predict(depth, mask)
        joints = recog.getJoints(depth, mask)
        
        sign = clf.predict([joints2dist(joints)])
        print sign.tolist()[0]
        #clientsocket.send(sign.tolist()[0])
        if (sign.tolist()[0] != actual_sign):
            if (sign.tolist()[0] == 'rest'):
                #open_hand(hand)
                actual_sign = sign.tolist()[0]
            elif (sign.tolist()[0] == 'v'):
                #perform_V(hand)
                actual_sign = sign.tolist()[0]
            elif (sign.tolist()[0] == 'w'):
                #perform_W(hand)
                actual_sign = sign.tolist()[0] 
            elif (sign.tolist()[0] == 's'):
                #perform_S(hand)
                actual_sign = sign.tolist()[0]     
        imshow("rgb", cvtColor(rgb, COLOR_RGB2BGR))
        imshow("prediction", prediction)


        waitKey(1)

        if not pos[0] and not pos[1] and not pos[2]:
            print ("Hand position lost. Exiting ...")
            break
