# EMG in imports
import numpy as np
import datetime as dt
from socket import timeout
import time
import serial

# Model imports
from tensorflow.keras import models, Model

# sevo imports
from board import SCL, SDA
import busio
from adafruit_pca9685 import PCA9685
from adafruit_servokit import ServoKit
import adafruit_motor.servo



### COLLECT DATA ###
SAMPLE_LENGTH = 150
CHANNELS = 12
COLLECT_DATA = True

# set up the serial line
PORT = '/dev/ttyACM0'
ser = serial.Serial(PORT, 115200, timeout=1)


# servo setup
i2c_bus = busio.I2C(SCL, SDA)
pca = PCA9685(i2c_bus)
kit = ServoKit(channels=16)

# Load Keras models
tt = models.load_model('b_final_thumb.keras')
it = models.load_model('b_final_index.keras')
mt = models.load_model('b_final_middle.keras')
rt = models.load_model('b_final_ring.keras')
pt = models.load_model('b_final_pinkey.keras')


def main():
    while True:
        # Data in
        rawEMG = getData()

        # Predictions generated
        posEsts = np.zeros((5,3))
        posEsts[0,:] = tt.predict(rawEMG)
        posEsts[1,:] = it.predict(rawEMG)
        posEsts[2,:] = mt.predict(rawEMG)
        posEsts[3,:] = rt.predict(rawEMG)
        posEsts[4,:] = pt.predict(rawEMG)

        # Coverted to coordinates
        pos = coordsToPos(posEsts)

        # Rendered in servo position
        servosTo(pos)

    # close serial
    ser.close()



### Helper Functions ###

def coordsToPos(coords):
    # Compute distance
    dist = np.linalg.norm(coords, axis=1).T

    # Normalize by 5th and 95th percentile of test data
    dMin = np.array([])
    dMax = np.array([])

    normDist = np.clip((dist-dMin)/(dMax-dMin), 0, 1)

    sMin = np.array([30,25,25,25,25])
    sMax = np.array([180,125,150,125,160])

    return normDist*(sMax-sMin)+sMin

def getData():
    # Create output array
    a = np.zeros((SAMPLE_LENGTH, CHANNELS))

    # Holds the previous timestamp
    priorTime = ""
    # Holds the summed data from those readings with timestamps identical to the last timestamp
    priorData = np.zeros(CHANNELS)
    # Counts the number of identical timestamps to average
    nPrior = 1
    # indexes the number of times filled
    n = 0

    while n < SAMPLE_LENGTH:
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
                floatData = [float(x) for x in dataSplit]
                priorData += np.array(floatData)
                nPrior += 1

            # Otherwise write to output file
            else:
                # write out
                if(nPrior >= 1):
                    # Compute average
                    averageData = priorData/nPrior
                    a[n,:] = averageData
                    n += 1

                # Convert data from current timestep
                priorTime = time
                floatData = [float(x) for x in dataSplit]
                priorData = np.array(floatData)
                nPrior = 1
    return a

def servosTo(pos, sleep=0.25):
    # Save the servos
    pos = np.clip(pos,10,180)
    # Move to position
    kit.servo[12].angle = pos[0]
    kit.servo[8].angle = pos[1]
    kit.servo[11].angle = pos[2]
    kit.servo[9].angle = pos[3]
    kit.servo[10].angle = pos[4]
    time.sleep(sleep)


if __name__ == '__main__':
    main()

