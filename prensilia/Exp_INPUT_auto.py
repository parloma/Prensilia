#Parameters: RF First Classification Layer - RF Second Classification Layer - Name of the volunteer
#Automatic -> NO sign recognition

#Import required
import sys
from os import path,sep,mkdir
import numpy as np
from cv2 import *
from hand_grabber import PyOpenNIHandGrabber
from pose_recognizer import PyPoseRecognizer
import xml.etree.ElementTree as ET
import Image
from random import *
import time
from my_fun import *
from sklearn.externals import joblib
from hand import *
import base64
import datetime
import socket
from Crypto.Cipher import AES # encryption library

# the character used for padding--with a block cipher such as AES, the value
# you encrypt must be a multiple of BLOCK_SIZE in length.  This character is
# used to ensure that your value is always a multiple of BLOCK_SIZE
PADDING = '{'
BLOCK_SIZE = 64

# one-liner to sufficiently pad the text to be encrypted
pad = lambda s: s + (BLOCK_SIZE - len(s) % BLOCK_SIZE) * PADDING

# one-liners to encrypt/encode and decrypt/decode a string
# encrypt with AES, encode with base64
EncodeAES = lambda c, s: base64.b64encode(c.encrypt(pad(s)))

#Parameters
WIDTH=640
HEIGHT=480
RADIUS = 150.0
USE_CPU = False
USE_PROB = False

#Joints Names and Signs List
JOINT_IDX2NAME = ["thumb_3_R", "thumb_2_R", "thumb_1_R",
                  "pinky_3_R", "pinky_2_R", "pinky_1_R",
                  "ring_3_R", "ring_2_R", "ring_1_R",
                  "middle_3_R", "middle_2_R", "middle_1_R",
                  "index_3_R", "index_2_R", "index_1_R",
                  "thumb_palm_R", "pinky_palm_R", "ring_palm_R", "middle_palm_R", "index_palm_R",
                  "palm_R", "wrist_R"]

SIGN_LIST = ['A','B','C','D','F','H','I','K','L','O','P2','S1','V','W','X','Y']
SIGN_INDEX = 0
SIGN_SIZE = 16
MAX_POSES = 160

#Communication Parameters
PASSCODE = 'PARLOMA3'*2
SIGN_WINDOW_NUMBER = 5
#IP = 'localhost'
#IP = '10.10.0.1'
#PORT = 8089
IP = '192.168.85.201'
PORT = 9091

#Class managing the Client Socket used to send messages to the Haptic Interface Controller (acting as a server)
class ClientSocket:
    def __init__(self, IP, PORT, PASSCODE, sock=None):
        if sock is None:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((IP, PORT))
        else:
            self.client_socket = sock

    def send_msg(self,msg):
        totalsent = 0
        while totalsent < len(msg):
            sent = self.client_socket.send(msg[totalsent:])
            if sent == 0:
                raise RuntimeError, "Connection to Server is DOWN!"
            totalsent = totalsent + sent

#Create Random List of Signs
def create_list():
    today = datetime.datetime.now()
    timems = today.strftime("%S")
    seed(10 + int(float(timems)))
    timem = today.strftime("%M")
    seed(7 + randint(1,3*int(float(timem))))
    sign_lista = SIGN_LIST*10
    lista = []
    M_POSES = 160
    while M_POSES>0:
        num = randint(0,MAX_POSES-1)
        lista.append(sign_lista[num])
        sign_lista[num] = sign_lista[M_POSES-1]
        M_POSES = M_POSES-1
    #print lista
    return lista
    
#main
if __name__=="__main__":
    
    if len(sys.argv)!=4:
        print("Usage: > python script_name forest1 forest2 person_name")

    else:

        if not path.exists(sys.argv[3]):
            mkdir(sys.argv[3])
            print "New folder created for this experiment"

        client = ClientSocket(IP, PORT, 'P'*16)

        #CV2 Windows
        namedWindow("rgb")
        namedWindow("masc")
        namedWindow("segno")
        #OpenNi tracker and Cypher Initialization
        #recog = PyPoseRecognizer(WIDTH, HEIGHT, sys.argv[1], USE_CPU, 320)
        #grabber = PyOpenNIHandGrabber()

        lista = create_list()
        clf = joblib.load(sys.argv[2])
        crypt = AES.new(PASSCODE)
        results = "Iteration \t Sign Required \t Sign recognized \n"

        out_file = open(sys.argv[3]+sep+"resultsInput.txt","w")
        out_file.write('Sign to perform \t Sign Recognized \n')
        out_file.close()

        out_file = open(sys.argv[3]+sep+"listaInput.txt","w")
        out_file.write(str(lista))
        out_file.close()

        #OpenNi wave to start tracking
        for i in range(0,MAX_POSES):
        #for i in range(0,10):
            #print("Wave the hand in front of the sensor \n")
            #while True:
                #rgb, depth = grabber.grabFrames()
                #pos = grabber.getHand3DPos() 
                #if len(pos) > 2:
                    #if pos[0] or pos[1] or pos[2]:
                        #break

            #print("+/-: increase/decrease the segmentation radius")
            #print("space: start/stop recording")
            #print("q: exit")

            #Sign to be performed - shown for 5 seconds ca. after tracker is acquired
            sign = lista[i]
            print("Perform sign %s\n"%sign)
            image = imread("."+sep+"signs"+sep+sign+".png")

            for j in range(0,50):
                imshow("segno", image)
                #rgb, depth = grabber.grabFrames()
                #pos = grabber.getHand3DPos()
                #mask = grabber.segment(depth, pos, RADIUS)
                #prediction = recog.predict(depth, mask)
                #joints = recog.getJoints(depth, mask)

                #signRecog = clf.predict([joints2dist(joints)])

                #imshow("rgb", cvtColor(rgb, COLOR_RGB2BGR))
                #imshow("masc", mask)
                k = waitKey(1)

            #if (USE_PROB):
                #prob = clf.predict_proba([joints2dist(joints)])
                #mm = prob.argmax()
                #signRecog = signs[mm]
                #prob = prob[0][mm]
            #else:
                #signRecog = clf.predict([joints2dist(joints)])
                #signRecog = signRecog.tolist()[0]
                #prob = 1

            signRecog = sign
            prob = 1

            if(prob <= 0.3):
                print "No sign recognized"
                signRecog = "NN"
                signRecogC = EncodeAES(" ")
            else:
            #Cypher
            #print signRecog
                signRecogC = EncodeAES(crypt,signRecog)
                #print len(signRecogC)

            #print sys.argv[2]+sep+"sign %d RGB"%i

            #Saving RGB and Segmented Depthmap for further references and result string for postprocessing
            #imwrite(sys.argv[3]+sep+"sign %d RGB.jpg"%i,cvtColor(rgb, COLOR_RGB2BGR))
            #imwrite(sys.argv[3]+sep+"sign %d.jpg"%i,mask)
            tmp = ""+str(i)+"\t"+sign+"\t"+signRecog+" \n"
            print tmp
            results = results + tmp

            #Send and acknowledge
            client.send_msg(signRecogC)
            print("Recognized and sent sign %s\n"%signRecog)
            #The hand will hold on the sign for 5 seconds
            time.sleep(8)
            #prediction = recog.predict(depth, mask)

            #At the end of each iteration save the results related to the volunteer
            results = results + "\n"

            out_file = open(sys.argv[3]+sep+"resultsInput.txt","a")
            out_file.write(sign + '\t' + signRecog + '\n')
            out_file.close()
