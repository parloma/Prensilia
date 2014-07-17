import serial
import time
import struct

#Class managing communication with the hand
class Hand:

	# ToDo: These should be constant values	
    THUMB = 0x01
    INDEX = 0x02
    MIDDLE = 0x03
    RING = 0x04
    LITTLE = 0x05
    THUMB_O = 0x06
    START_BIT = 0b10000000
    START_BIT_GET = 0b01000101
    #Middle stands for 'in the middle btw open and close', does NOT refer to the middle finger
    START_BIT_MIDDLE = 0b01000100    
    
    finger_set = {'thumb':THUMB, 'index':INDEX, 'middle':MIDDLE, 'ring':RING, 'little':LITTLE, 'thumb_o': THUMB_O}
    command_set = {'open':0b01000000, 'close':0, 'middle1':0b10000000, 'middle2':0b01000000, 'middle3':0b11000000, 'middle4':0b10111000, 'middle5':0b01110000, 'middle6':0b01001000, 'middle_close':0b11111111, 'middle_open':0b00000000}
	
	# Contructor
    def __init__(self, hand_port):
        #self.serial_comm = 0
        self.hand_port = hand_port
        self.serial_comm = serial.Serial(port=hand_port, baudrate=115200)
        if (self.serial_comm.isOpen()):
            print 'Communication was already opened!'
        else:
            self.serial_comm.open();
            if (self.serial_comm.isOpen()):
                print 'Communication Opened!'
            else:
                print 'Error: impossible to establish communication!'
	
	# Destructor
    def __del__(self):
        self.serial_comm.close();
        print 'Communication Closed!'

    #Main communication functions
    def send_command(self, cmd):
        self.serial_comm.write(chr(cmd))

    def send_commands(self, cmds):
        cmds2send = ''
        for cmd in cmds:
            cmds2send += chr(cmd)
        self.serial_comm.write(cmds2send)

    def move_finger(self, finger, command, vel):
        if (finger in self.finger_set and command in self.command_set and vel > 0 and vel <= 0b1111111111):
            cmds = [0]*2
            cmds[0] = self.START_BIT + self.command_set[command] + self.finger_set[finger]*4 + vel/256
            cmds[1] = vel%256
            self.send_commands(cmds)

    def move_finger_middle(self, finger, pos):
        if (finger in self.finger_set and pos in self.command_set):
            cmds = [0]*3
            cmds[0] = self.START_BIT_MIDDLE
            cmds[1] = self.finger_set[finger]
            cmds[2] = self.command_set[pos]
            self.send_commands(cmds)


    #Preset Perform Action
    def perform_rest(self):
        self.send_command(76)
        #time.sleep(1)
        self.move_finger_middle('thumb_o', 'middle_open')

    def perform_softCalibration(self):
        self.send_command(70)

    def perform_hardCalibration(self):
        self.send_command(66)

    def perform_a(self):
        self.move_finger_middle('index', 'middle_close')
        self.move_finger_middle('middle', 'middle_close')
        self.move_finger_middle('ring', 'middle_close')
        self.move_finger_middle('little', 'middle_close')
        time.sleep(1)
        self.move_finger_middle('thumb', 'middle_close')
        self.move_finger_middle('thumb_o', 'middle_open')

    def perform_b(self):
        self.move_finger_middle('thumb', 'middle_close')
        self.move_finger_middle('thumb_o', 'middle_close')
        self.move_finger_middle('index', 'middle_open')
        self.move_finger_middle('middle', 'middle_open')
        self.move_finger_middle('ring', 'middle_open')
        self.move_finger_middle('little', 'middle_open')

    def perform_c(self):
        self.move_finger_middle('thumb', 'middle1')
        self.move_finger_middle('thumb_o', 'middle2')
        self.move_finger_middle('index', 'middle1')
        self.move_finger_middle('middle', 'middle1')
        self.move_finger_middle('ring', 'middle1')
        self.move_finger_middle('little', 'middle6')

    def perform_d(self):
        self.move_finger_middle('thumb', 'middle_close')
        self.move_finger_middle('thumb_o', 'middle_close')
        self.move_finger_middle('index', 'middle_open')
        self.move_finger_middle('middle', 'middle_close')
        self.move_finger_middle('ring', 'middle_close')
        self.move_finger_middle('little', 'middle_close')

    def perform_f(self):
        self.move_finger_middle('thumb', 'middle4')
        self.move_finger_middle('thumb_o', 'middle4')
        self.move_finger_middle('index', 'middle_close')
        self.move_finger_middle('middle', 'middle_open')
        self.move_finger_middle('ring', 'middle_open')
        self.move_finger_middle('little', 'middle_open')

    def perform_h(self):
        self.move_finger_middle('index', 'middle2')
        self.move_finger_middle('middle', 'middle1')
        self.move_finger_middle('ring', 'middle_close')
        self.move_finger_middle('little', 'middle_close')
        time.sleep(1)
        self.move_finger_middle('thumb', 'middle_close')
        self.move_finger_middle('thumb_o', 'middle_close')

    def perform_i(self):
        self.move_finger_middle('index', 'middle_close')
        self.move_finger_middle('middle', 'middle_close')
        self.move_finger_middle('ring', 'middle_close')
        self.move_finger_middle('little', 'middle_open')
        time.sleep(1)
        self.move_finger_middle('thumb', 'middle_close')
        self.move_finger_middle('thumb_o', 'middle_close')

    def perform_k(self):
        self.move_finger_middle('index', 'middle_open')
        self.move_finger_middle('middle', 'middle4')
        self.move_finger_middle('ring', 'middle_close')
        self.move_finger_middle('little', 'middle_close')
        time.sleep(1)
        self.move_finger_middle('thumb', 'middle_close')
        self.move_finger_middle('thumb_o', 'middle_open')

    def perform_l(self):
        self.move_finger_middle('thumb', 'middle_open')
        self.move_finger_middle('thumb_o', 'middle_open')
        self.move_finger_middle('index', 'middle_open')
        self.move_finger_middle('middle', 'middle_close')
        self.move_finger_middle('ring', 'middle_close')
        self.move_finger_middle('little', 'middle_close')

    def perform_o(self):
        self.move_finger_middle('thumb', 'middle3')
        self.move_finger_middle('thumb_o', 'middle1')
        self.move_finger_middle('index', 'middle4')
        self.move_finger_middle('middle', 'middle1')
        self.move_finger_middle('ring', 'middle1')
        self.move_finger_middle('little', 'middle6')

    def perform_p2(self):
        self.move_finger_middle('thumb', 'middle_close')
        self.move_finger_middle('thumb_o', 'middle_close')
        time.sleep(1)
        self.move_finger_middle('index', 'middle1')
        self.move_finger_middle('middle', 'middle3')
        self.move_finger_middle('ring', 'middle_close')
        self.move_finger_middle('little', 'middle_close')

    def perform_s1(self):
        self.move_finger_middle('thumb', 'middle_open')
        self.move_finger_middle('thumb_o', 'middle_open')
        self.move_finger_middle('index', 'middle_close')
        self.move_finger_middle('middle', 'middle_close')
        self.move_finger_middle('ring', 'middle_close')
        self.move_finger_middle('little', 'middle_close')
			
    def perform_v(self):
        self.move_finger_middle('thumb', 'middle_close')
        self.move_finger_middle('thumb_o', 'middle_close')
        self.move_finger_middle('index', 'middle_open')
        self.move_finger_middle('middle', 'middle_open')
        self.move_finger_middle('ring', 'middle_close')
        self.move_finger_middle('little', 'middle_close')

    def perform_w(self):
        self.move_finger_middle('thumb', 'middle_close')
        self.move_finger_middle('thumb_o', 'middle_close')
        self.move_finger_middle('index', 'middle_open')
        self.move_finger_middle('middle', 'middle_open')
        self.move_finger_middle('ring', 'middle_open')
        self.move_finger_middle('little', 'middle_close')

    def perform_x(self):
        self.move_finger_middle('index', 'middle5')
        self.move_finger_middle('middle', 'middle_close')
        self.move_finger_middle('ring', 'middle_close')
        self.move_finger_middle('little', 'middle_close')
        time.sleep(1)
        self.move_finger_middle('thumb', 'middle_close')
        self.move_finger_middle('thumb_o', 'middle_close')

    def perform_y(self):
        self.move_finger_middle('thumb', 'middle_open')
        self.move_finger_middle('thumb_o', 'middle_open')
        self.move_finger_middle('index', 'middle_close')
        self.move_finger_middle('middle', 'middle_close')
        self.move_finger_middle('ring', 'middle_close')
        self.move_finger_middle('little', 'middle_open')

    def perform_rest2(self):
        self.move_finger_middle('thumb', 'middle_open')
        self.move_finger_middle('thumb_o', 'middle_open')
        self.move_finger_middle('index', 'middle_open')
        self.move_finger_middle('middle', 'middle_open')
        self.move_finger_middle('ring', 'middle_open')
        self.move_finger_middle('little', 'middle_open')


    SIGN_LIST = ['A','B','C','D','F','H','I','K','L','O','P2','S1','V','W','X','Y','REST']
    FNCT_LIST = {'A':perform_a, 'B':perform_b, 'C':perform_c, 'D':perform_d, 'F':perform_f, 'H':perform_h, 'I':perform_i, 'K':perform_k, 'L':perform_l, 'O':perform_o, 'P2':perform_p2, 'S1':perform_s1, 'V':perform_v, 'W':perform_w, 'X':perform_x, 'Y':perform_y, 'REST':perform_rest2}

    def get_sign(self):
        res = ''
        for finger in ['middle', 'ring', 'little', 'thumb', 'thumb_o']:
            #time.sleep(1)            
            cmds = [0]*2
            cmds[0] = self.START_BIT_GET
            cmds[1] = self.finger_set[finger]
            self.send_commands(cmds)
            #time.sleep(1)
            #print self.serial_comm.read() + ' munnizza'
            value = ord(self.serial_comm.read())
            #print finger
            #print str(value)
            #res = res + ' ' + self.serial_comm.read()
            res = res + str(value) + ' '
        return res


    def perform_sign(self, msg):
        print 'Segno da stampare ' + msg
        self.FNCT_LIST[msg](self)
        time.sleep(5)
        return self.get_sign()
        

if __name__=="__main__":
    hand = Hand('/dev/ttyUSB0')
    
    #hand.perform_hardCalibration()
    #time.sleep(60)
    #hand.perform_rest()
    #time.sleep(2)
    #hand.FNCT_LIST['H'](hand)

    #time.sleep(5)
    #print hand.perform_sign('REST')

    #for msg in hand.SIGN_LIST:
    #for msg in ['D','F','H']:
    for msg in ['REST']:
        hand.perform_rest()
        print 'Performin Sign ' + msg
        time.sleep(2)
        hand.FNCT_LIST[msg](hand)
        time.sleep(5)
        hand.get_sign()
