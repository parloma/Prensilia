from cv2 import namedWindow, imshow, waitKey, cvtColor, COLOR_RGB2BGR
from hand_grabber import PyOpenNIHandGrabber
from pose_recognizer import PyPoseRecognizer
from my_fun import *
from sklearn.externals import joblib
from SerialClient import *
import time



WIDTH = 640
HEIGHT = 480
USE_CPU = False
USE_FORESTBIG = True
USE_PROB = True


if __name__=="__main__":

    raspberryClient = ClientSocket('10.10.0.1', 9091, '/dev/ttyUSB0', 'P'*16)
        
    sign_set = {'s':[72,   5, 250, 250, 250, 250,   5], 'v':[72, 250,   5,   5, 250, 250,   5], 'u':[72, 250,   5,   5, 250, 250,   5], 'w':[72, 250,   5,   5,   5, 250,   5], 'rest':[ 72,  5,   5,   5,   5,   5,   5], 'f':[72, 200, 160,5,5,5,120]}

    signs = ['f', 'horns', 'rest', 's', 'u', 'v', 'w']

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

    print("Wave the hand in front of the sensor")

    while True:
        while True:
            rgb, depth = grabber.grabFrames()
            pos = grabber.getHand3DPos()     
            if pos[0] or pos[1] or pos[2]:
                break


        print("CTRL+C to exit")
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
                print "No sign recognized"
                sign_counter = 0
                actual_sign = ""
            else: 
                print sign
                if (sign != actual_sign):
                    if (sign == new_sign):
                        sign_counter += 1
                        if (sign_counter == 5):
                            if (sign in sign_set):
                                raspberryClient.send_msg(''.join([chr(i) for i in sign_set[sign]]),0)
                                actual_sign = sign
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
