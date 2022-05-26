import numpy as np
import datetime as dt
from socket import timeout
import time
import serial
from collections import deque
import queue

CHANNELS = 12
COLLECT_DATA = True

# set up the serial line
ser = serial.Serial('COM7', 115200, timeout=1)
time.sleep(2)


# Create file
t = dt.datetime.now()
s = t.strftime("%b.%d.%H.%M.%S")
file = open("EMG_"+s+".txt", "w+")


# Holds the previous timestamp
priorTime = ""
# Holds the summed data from those readings with timestamps identical to the last timestamp
priorData = np.zeros(CHANNELS)
# Counts the number of identical timestamps to average
nPrior = 1
n = 0

while COLLECT_DATA:
    n += 1
    print(n)
    # Get timestamp
    t = dt.datetime.now()
    time = t.strftime("%H:%M:%S.%f")

    # Read the serial line
    data = ser.readline().decode("utf-8")        # read a byte string, decode byte string into Unicode 
    data = data.rstrip() # remove \n and \r
    dataSplit = data.split(",")

    # Ensure a good read
    if (len(dataSplit) == CHANNELS) and not ("" in dataSplit) and not  (" " in dataSplit):
         # Now function does not work properly; if times identical, average
        if(time == priorTime):
            print("True")
            floatData = [float(x) for x in dataSplit]
            priorData += np.array(floatData)
            nPrior += 1

        # Otherwise write to output file
        else:
            # write out
            if(nPrior >= 1):
                # Compute average
                averageData = priorData/nPrior
                averageData = ''.join([str(num) +", " for num in averageData])
                # Write out
                file.write(priorTime + ", " + averageData)
                file.write('\n')

            # Convert data from current timestep
            priorTime = time
            floatData = [float(x) for x in dataSplit]
            priorData = np.array(floatData)
            nPrior = 1

ser.close()