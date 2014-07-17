#Parameters: RF First Classification Layer - RF Second Classification Layer - Name of the volunteer

#Import required
import sys
from os import mkdir,sep,path
#import numpy as np
#from cv2 import *
#from hand_grabber import PyOpenNIHandGrabber
#from pose_recognizer import PyPoseRecognizer
import thread
import xml.etree.ElementTree as ET
#import Image
from random import *
import time
#from my_fun import *
#from sklearn.externals import joblib
from robot_hand import *
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
DecodeAES = lambda c, e: c.decrypt(base64.b64decode(e)).rstrip(PADDING)

SIGN_LIST = ['A','B','C','D','F','H','I','K','L','O','P2','S1','V','W','X','Y']

SIGN_INDEX = 0
SIGN_SIZE = 16
MAX_POSES = 100

#Communication Parameters
PASSCODE = 'PARLOMA3'*2
SIGN_WINDOW_NUMBER = 5
#IP = 'localhost'
#IP = '10.10.0.1'
IP = '192.168.85.201'
#PORT = 8089
PORT = 9091
MSGLEN = 88

class ServerSocket:
    def __init__(self, IP, PORT, PASSCODE, ser, name):
        self.server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.server_socket.bind((IP,PORT))
        self.server_socket.listen(1)

        self.hand = Hand(ser)
        #self.hand.perform_hardCalibration()
        self.hand.perform_softCalibration()
        self.hand.perform_rest()
        #self.hand = 0
        self.name = name

    def start(self, crypt):
        print "Initializing..."
        #Initializing -> strangely the first sign won't be performed
        res = self.hand.perform_sign('A')
        res = self.hand.perform_sign('REST')
        print 'Ready! Waiting on IP '+IP+' and PORT '+str(PORT)
        #return
        while True:
            client_socket, address = self.server_socket.accept();
            print 'Listening to client, address:'
            print address
            thread.start_new_thread(self.handController, (self.hand, crypt, client_socket, address))
        
        
    def handController(self, hand, crypt, client_socket, address, *args):
        #actual_sign = 'rest'
        #actual_counter = 0

        while True:
            msg = ''
            while len(msg) < MSGLEN:
                chunk = client_socket.recv(MSGLEN-len(msg))
                if chunk == '':
                    print "Connection to Client is DOWN!"
                    print address
                    client_socket.close()
                    return
                msg = msg + chunk
            buf = msg

            if len(buf) != MSGLEN: # client closed or network error
                print 'Client Closed or Communication Error'
                print address
                client_socket.close()
                return
            else:
                buf = DecodeAES(crypt, buf)
                print buf + ' RECEIVED'

                if buf == 'quit':
                    print 'Ok, Quitting'
                    return
                else:
                    x = buf in SIGN_LIST
                    if x == False:
                        print 'Invalid sign received'
                        out_file = open(self.name+sep+"resultsPerformedHand.txt","a")
                        out_file.write('Invalid sign ' + buf + ' received! \n')
                        out_file.close()
                    else:
                        res = hand.perform_sign(buf)
                        #time.sleep(4)
                        hand.perform_rest()

                        out_file = open(self.name+sep+"resultsPerformedHand.txt","a")
                        out_file.write(res + '\n')
                        out_file.close()

                    out_file = open(self.name+sep+"resultsReceivedInternet.txt","a")
                    out_file.write(buf + '\t' + buf + '\n')
                    out_file.close()
                    
                    #if actual_sign == buf:
                        #actual_counter += 1
#                       if actual_counter == SIGN_WINDOW_NUMBER:
#                           hand.perform_sign(buf)
#                           print 'Sign Performed'
                    #else:
                        #actual_sign = buf
                        #actual_counter = 1

#main
if __name__=="__main__":
    
    if len(sys.argv)!=3:
        print("Usage:Client > python script_name serial volunteer")

    else:

        if not path.exists(sys.argv[2]):
            mkdir(sys.argv[2])
            print "New folder created for this experiment"

        out_file = open(sys.argv[2]+sep+"resultsPerformedHand.txt","w")
        out_file.write('#Reference pose of PRENSILIA Hand wrt specified poses \n')
        out_file.write('#Joints order middle, ring, little, thumb, thumb_o \n')
        out_file.close()

        out_file = open(sys.argv[2]+sep+"resultsReceivedInternet.txt","w")
        out_file.write('Sign received from internet' + '\t' + 'Actual joints positions' + '\n')
        out_file.close()			

        server = ServerSocket(IP, PORT, 'P'*16, sys.argv[1], sys.argv[2])
        crypt = AES.new(PASSCODE)
        server.start(crypt)

        #while True:
            # Accept and dispatch connection from client
            #print 'Waiting on IP '+IP+' and PORT '+str(PORT)
            #(SocketClient, address) = server.server_socket.accept()
            #handController(SocketClient, address, crypt)
