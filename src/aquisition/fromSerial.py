import datetime as dt
from socket import timeout
import time
import serial

# The first line of data is not read properly
FIRST = True

# set up the serial line
ser = serial.Serial('COM10', 115200, timeout=1)
time.sleep(2)


# Create file
t = dt.datetime.now()
s = t.strftime("%b.%d.%H.%M.%S")
file = open("EMG_"+s+".txt", "w+")

while True:
    # Get timestamp
    t = dt.datetime.now()
    line = t.strftime("%H:%M:%S.%f")

    # Read the serial line
    b = ser.readline()         # read a byte string
    string_n = b.decode("utf-8")  # decode byte string into Unicode 
    string = string_n.rstrip() # remove \n and \r

    if not FIRST:
        # Compose output
        line = line + ", " + string
        file.write(line)
        file.write('\n')
    # Reset flag
    FIRST = False

ser.close()