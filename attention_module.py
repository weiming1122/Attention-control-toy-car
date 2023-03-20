
from threading import Timer
import time

import sys
import struct
import numpy as np
import atexit
import datetime
import glob
import serial
from serial import Serial

# Define variables
SAMPLE_RATE = 250.0  # Hz

PACKET_START_BYTE1 = 0xAA  # start of data packet
PACKET_START_BYTE2 = 0x77

# packet type
PACKET_TYPE_RAW_DATA = 0x10
PACKET_TYPE_ATTENTION = 0x11


# Sample number
EEG_SAMPLE_NUM = 8

class AttentionModule(object):
    '''
    
    '''
    def __init__(self, port, baud=115200, timeout=None, max_packets_skipped=1):
        self.baud = baud
        self.timeout = timeout
        self.max_packets_skipped = max_packets_skipped
        self.port = port
        self.start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

        # Connecting to the board
        self.ser = Serial(port=self.port, baudrate=self.baud, timeout=self.timeout)

        print('Serial established')
        # Perform a soft reset of the board
        time.sleep(2)

        self.packets_dropped = 0
        self.packets_type = 0
        self.read_state = 0
        self.streaming = False
    
    def parse_board_data(self, maxbytes2skip=3000):
        """Parses the data from the Cyton board into an OpenBCISample object."""
        def read_board(n):
            bb = self.ser.read(n)
            if not bb:
                print('Device appears to be stalling. Quitting...')
                sys.exit()
                raise Exception('Device Stalled')
                sys.exit()
                return '\xFF'
            else:
                return bb

        for rep in range(maxbytes2skip):

            # Start Byte 
            if self.read_state == 0:
                b1 = read_board(1)
                if struct.unpack('B', b1)[0] == PACKET_START_BYTE1:
                    b2 = read_board(1)
                    if struct.unpack('B', b2)[0] == PACKET_START_BYTE2:
                        if rep != 0:
                            print('Skipped %d bytes before start found' % rep)
                            rep = 0
                        self.read_state = 1

            # Device
            elif self.read_state == 1:
                b = read_board(1)
                packet_device = struct.unpack('B', b)[0]
                self.read_state = 2

            # ID
            elif self.read_state == 2:
                b = read_board(1)
                packet_id = struct.unpack('B', b)[0]
                self.read_state = 3

            # Type
            elif self.read_state == 3:
                channels_data = []
                band_power_data = []
                b = read_board(1)
                self.packets_type = struct.unpack('B', b)[0]
                if self.packets_type == PACKET_TYPE_RAW_DATA:
                    self.read_state = 4
                elif self.packets_type == PACKET_TYPE_ATTENTION:
                    self.read_state = 5
                else:
                    self.read_state = 0

            # Raw data
            elif self.read_state == 4:
                for c in range(EEG_SAMPLE_NUM):
                    # # Read 3 byte integers
                    # literal_read = read_board(3)
                    # unpacked = struct.unpack('3B', literal_read)

                    # # Translate 3 byte int into 2s complement
                    # if unpacked[0] > 127:
                    #     pre_fix = bytes(bytearray.fromhex('FF'))
                    # else:
                    #     pre_fix = bytes(bytearray.fromhex('00'))
                    # literal_read = pre_fix + literal_read
                    # myInt = struct.unpack('>i', literal_read)[0]

                    # Append channel to channels data
                    literal_read = read_board(4)
                    myFloat = struct.unpack('<f', literal_read)[0]
                    channels_data.append(myFloat)

                self.read_state = 6
                
            # Attention
            elif self.read_state == 5:
                for c in range(4):
                    literal_read = read_board(4)
                    myFloat = struct.unpack('<f', literal_read)[0]
                    band_power_data.append(myFloat)
                self.read_state = 6
            
            # CRC 8
            elif self.read_state == 6:
                read_board(1)
                
                self.read_state = 0
                # if self.packets_type == PACKET_TYPE_RAW_DATA:
                sample = {'channels_data': channels_data, 'band_power': band_power_data}
                return sample

    def write_command(self, command):
            self.ser.write(command.encode())
            time.sleep(0.1)

    def start_stream(self, callback):
        """Start handling streaming data from the board. Call a provided callback for every single sample that is processed."""
        if not self.streaming:
            # self.write_command('AT+SLAVE=0\r\n')
            self.streaming = True

        # Enclose callback function in a list
        if not isinstance(callback, list):
            callback = [callback]

        while self.streaming:
            #read current sample
            sample = self.parse_board_data()
            if sample:
                for call in callback:
                    call(sample)