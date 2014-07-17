import serial

def start_communication():
    hand = serial.Serial('/dev/ttyUSB0', 115200)
    return hand
    
def close_communication(hand):
    hand.close()


def control_thumb(hand, command):
    if (command == 'open'):
        hand.write(chr(196+2)+chr(255)) # open Thumb
    elif (command == 'close'):
        hand.write(chr(132+2)+chr(255)) # close Thumb

def control_index(hand, command):
    if (command == 'open'):
        hand.write(chr(202)+chr(255)) # open Thumb
    elif (command == 'close'):
        hand.write(chr(138)+chr(255)) # close Thumb
        
def control_middle(hand, command):
    if (command == 'open'):
        hand.write(chr(204+2)+chr(255)) # open Thumb
    elif (command == 'close'):
        hand.write(chr(140+2)+chr(255)) # close Thumb
        
def control_ring(hand, command):
    if (command == 'open'):
        hand.write(chr(208+2)+chr(255)) # open Thumb
    elif (command == 'close'):
        hand.write(chr(144+2)+chr(255)) # close Thumb        
        
def control_thumb_O(hand, command):
    if (command == 'open'):
        hand.write(chr(216+2)+chr(255)) # open Thumb
    elif (command == 'close'):
        hand.write(chr(152+2)+chr(255)) # close Thumb        
        
def control_little(hand, command):
    if (command == 'open'):
        hand.write(chr(212+2)+chr(255)) # open Thumb
    elif (command == 'close'):
        hand.write(chr(148+2)+chr(255)) # close Thumb    
        
def open_hand(hand):
    control_ring(hand, 'open')
    control_little(hand, 'open')
    control_index(hand, 'open')
    control_middle(hand, 'open')
    control_thumb(hand, 'open')
    control_thumb_O(hand, 'open')

def perform_V(hand):
    control_ring(hand, 'close')
    control_little(hand, 'close')
    control_index(hand, 'open')
    control_middle(hand, 'open')
    control_thumb(hand, 'close')
    control_thumb_O(hand, 'close')

def perform_W(hand):
    control_ring(hand, 'open')
    control_little(hand, 'close')
    control_index(hand, 'open')
    control_middle(hand, 'open')
    control_thumb(hand, 'close')
    control_thumb_O(hand, 'close')
            
def perform_S(hand):
    control_ring(hand, 'close')
    control_little(hand, 'close')
    control_index(hand, 'close')
    control_middle(hand, 'close')
    control_thumb(hand, 'open')
    control_thumb_O(hand, 'open')
