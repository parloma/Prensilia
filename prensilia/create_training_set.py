#Script to create Training Set for the Second Layer of Classification
#Parameters: Random Forest (First Layer of Classification) and Name of the Volunteer
#Except dynamic sings (G J Z from Italian tLIS)

import sys
#from os import sep, mkdir
from os import *
import numpy as np
from cv2 import *
from hand_grabber import PyOpenNIHandGrabber
from pose_recognizer import PyPoseRecognizer
import xml.etree.ElementTree as ET
import Image

WIDTH=640
HEIGHT=480
RADIUS = 150.0
USE_CPU = False

JOINT_IDX2NAME = ["thumb_3_R", "thumb_2_R", "thumb_1_R",
                  "pinky_3_R", "pinky_2_R", "pinky_1_R",
                  "ring_3_R", "ring_2_R", "ring_1_R",
                  "middle_3_R", "middle_2_R", "middle_1_R",
                  "index_3_R", "index_2_R", "index_1_R",
                  "thumb_palm_R", "pinky_palm_R", "ring_palm_R", "middle_palm_R", "index_palm_R",
                  "palm_R", "wrist_R"]

SIGN_LIST = ['A','B','C','D','E','F','H','I','K','L','M','N','O','P1','P2','Q','R','S1','S2','T','U','V','W','X','Y']
SIGN_INDEX = 0
SIGN_SIZE = 25
MAX_POSES = 100

if __name__=="__main__":
    
    if len(sys.argv)!=3:
        print("Usage: > python script_name forest person_name")

    if len(sys.argv)==3:

        if not path.exists(sys.argv[2]):
            mkdir(sys.argv[2])
            print "New folder created for this experiment"

        namedWindow("rgb")
        namedWindow("mask")
        namedWindow("sign")
    
        recog = PyPoseRecognizer(WIDTH, HEIGHT,
                                 sys.argv[1],
                                 USE_CPU, 320)
        grabber = PyOpenNIHandGrabber()
    
        while True:
            print("Wave the hand in front of the sensor\a")
            while True:
                rgb, depth = grabber.grabFrames()
                pos = grabber.getHand3DPos() 
                if len(pos) > 2:
                    if pos[0] or pos[1] or pos[2]:
                        break

            print("+/-: increase/decrease the segmentation radius")
            print("space: start/stop recording")
            print("q: exit")
            
            #Iterating for all the signs in the sign folder
            #Not recording unless when decided by the user - supervisor of the experiment
            recording = False
            if SIGN_INDEX < SIGN_SIZE:
                sign = SIGN_LIST[SIGN_INDEX]
                joints = []
                weights = []
                depthmaps = []
                #Showing visually the sign to perform to reduce errors rate
                image = imread("."+sep+"signs"+sep+sign+".png")
                #image = Image.open("."+sep+"signs"+sep+sign+".png").show(fig)
                #print("."+sep+"signs"+sep+sign+".png\a")
                imshow("sign", image)

                print("Perform sign %s, press spase to start/stop, q to exit from the sign\a" %sign)
                while True:
                    rgb, depth = grabber.grabFrames()

                    pos = grabber.getHand3DPos()
                    mask = grabber.segment(depth, pos, RADIUS)

                    currJoints = recog.getJoints(depth, mask)
                    
                    imshow("rgb", cvtColor(rgb, COLOR_RGB2BGR))
                    imshow("mask", mask)
                    
                    #Space to start recording for MAX_POSES consecutive frame
                    #Record a comprehensive set of depthmaps to compensate for feature dependencies (rotations, scale, distance ...)
                    k = waitKey(1)
                    if k != -1:
                        k %= 256
                        if chr(k) == '+':
                            RADIUS += 1.0
                        elif chr(k) == '-':
                            RADIUS -= 1.0
                        elif chr(k) == ' ':
                            recording = not recording
                            if (recording):
                                print 'Start Recording'
                                TMPjoints = []
                                TMPdepthmaps = []
                                POSES = 0
                            else:
                                print 'Stop Recording'
                        elif chr(k) == 'q':
                            break
                
                    if recording and POSES < MAX_POSES:
                        POSES = POSES + 1
                        TMPjoints.append(currJoints)
                        TMPdepthmaps.append(depth.copy())
                        print len(TMPjoints)

                    if recording and POSES >= MAX_POSES:
                        print "MAX POSES STORED"                  

                #User - Supervisor may decide if saving the single sign or to repeat it (in case of errors)
                print("Do you want to save sign? y/n\a")
                while True:
                    k = waitKey(1)
                    if k != -1:
                        k %= 256
                        if chr(k) == 'y':
                            SIGN_INDEX = SIGN_INDEX+1
                            joints = joints + TMPjoints
                            depthmaps = depthmaps + TMPdepthmaps
                            frameIdx = 0
                            mkdir(sys.argv[2]+sep+sign+sep)
                            for currJoints, depthmap in zip(joints, depthmaps):
                                #Creating the XML Tree to be used by the classifier
                                root = ET.Element('joints')
                                tree = ET.ElementTree(root)
                                jointIdx = 0
                                for joint in zip(currJoints):
                                    print joint, joint[0]
                                    jointNode = ET.SubElement(root, JOINT_IDX2NAME[jointIdx])
                                    coordsNode = ET.SubElement(jointNode, "coords")
                                    coordsNode.text = "%.3f %.3f %.3f"%(joint[0][0], joint[0][1], joint[0][2])
                                    #weightNode = ET.SubElement(jointNode, "weight")
                                    #weightNode.text = str(weight[0])
                                    jointIdx += 1
                                tree.write(sys.argv[2]+sep+sign+sep+"frame%d_joints.xml"%frameIdx)
                                imwrite(sys.argv[2]+sep+sign+sep+"frame%d_depth.png"%frameIdx, depthmap)
                                frameIdx += 1
                            break
                        else:
                            break
            else:
                print("Arrivederci e grazie di tutto il pesce\a")
                break                     
