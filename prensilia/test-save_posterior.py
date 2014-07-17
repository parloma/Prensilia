#PARLOMA testing script / possible saving a posteriori probability

import sys
import numpy as np
#from cv2 import namedWindow, imshow, waitKey, cvtColor, COLOR_RGB2BGR, GaussianBlur
from cv2 import *
from hand_grabber import PyOpenNIHandGrabber
from pose_recognizer import PyPoseRecognizer

N_LABELS = 22
WIDTH = 640
HEIGHT = 480
USE_CPU = False

DEFAULT_POSTERIOR = 0.5

if __name__=="__main__":

    #if len(sys.argv)==2 and int(sys.argv[1])>=0 and int(sys.argv[1])<22:

        #label = int(sys.argv[1])

        namedWindow("rgb")
        namedWindow("prediction")
        namedWindow("posterior")

        grabber = PyOpenNIHandGrabber()
        streamWidth = WIDTH
        streamHeight = HEIGHT
        trainingSetSamplesWidth = 320
        radius = 140

        recog = PyPoseRecognizer(streamWidth, streamHeight,
                                 "forest.xml",
                                 USE_CPU, trainingSetSamplesWidth)
        posteriorsThr = np.array([DEFAULT_POSTERIOR]*N_LABELS, dtype=np.float32)
        recog.setPosteriorThresholds(posteriorsThr)

        print("Wave the hand in front of the sensor")

        while True:
            rgb, depth = grabber.grabFrames()
            pos = grabber.getHand3DPos()     
            if pos[0] or pos[1] or pos[2]:
                break

        print("CTRL+C to exit")

        savePosterior=False
        #posteriors = []

        while True:
            rgb, depth = grabber.grabFrames()

            pos = grabber.getHand3DPos()
            mask = grabber.segment(depth, pos, radius)
            prediction = recog.predict(depth, mask)
            posterior = recog.posterior(depth, mask)

            #posterior = posterior[label]
            #posterior = GaussianBlur(posterior, (11,11), 5)

            imshow("rgb", cvtColor(rgb, COLOR_RGB2BGR))
            imshow("prediction", prediction)
            #imshow("posterior", posterior)

            k = waitKey(1)%256

            if k!=-1:
                if chr(k)=='q':
                    break
                elif chr(k)==' ':
                    savePosterior = not savePosterior

            if savePosterior:
                #posteriors.append(posterior)

                for i in range(22):
                    imwrite("tmp/posterior%d.png"%i, posterior[i]*255)
                    imwrite("tmp/depthmap.png", depth*(mask/255))
                '''
                i=0
                for posterior in posteriors:
                    prefix = "0"*(len(str(len(posteriors)))-len(str(i)))
                    imwrite("tmp/posterior%s%d.png"%(prefix, i), posterior*255)
                    i+=1
                '''
