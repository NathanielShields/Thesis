import board, busio, time, datetime as dt
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn


# connections to sensors
i2c = busio.I2C(board.SCL, board.SDA)

ads0 = ADS.ADS1115(i2c,address = 0x48, data_rate=860)
ads1 = ADS.ADS1115(i2c,address = 0x49, data_rate=860)
ads2 = ADS.ADS1115(i2c,address = 0x4B, data_rate=860)

chans = []

chans.append(AnalogIn(ads0, ADS.P0))
chans.append(AnalogIn(ads0, ADS.P1))
chans.append(AnalogIn(ads0, ADS.P2))
chans.append(AnalogIn(ads0, ADS.P3))
chans.append(AnalogIn(ads1, ADS.P0))
chans.append(AnalogIn(ads1, ADS.P1))
chans.append(AnalogIn(ads1, ADS.P2))
chans.append(AnalogIn(ads1, ADS.P3))
chans.append(AnalogIn(ads2, ADS.P0))
chans.append(AnalogIn(ads2, ADS.P1))
chans.append(AnalogIn(ads2, ADS.P2))
chans.append(AnalogIn(ads2, ADS.P3))

# Create file
t = dt.datetime.now()
s = t.strftime("%b.%d.%H:%M:%S")
file = open("EMG_"+s+".txt", "w+")


# Data Collection
period = 0.05
nt = time.time()

while True:
    nt+=period
    t = dt.datetime.now()
    line = t.strftime("%H:%M:%S.%f")

    for chan in chans:
        line+= "," + str(chan.value)
    
    file.write(line)
    file.write('\n')
    time.sleep(max(0,nt-time.time()))