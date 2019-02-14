import time
import datetime
from centerfinder import util
from centerfinder import sky

filename = '../data/SignalN3_mid.txt'
data = util.load_data(filename)
sk = sky.Sky(data, avg_bins=5)
start = time.time()
sk.vote2(108)
end = time.time()
with open('../log/time_log.txt', 'a') as f:
    f.write(str(datetime.datetime.now()))
    f.write('\n')
    f.write(filename)
    f.write('\n')
    f.write(str(sk))
    f.write('\n')
    f.write('Time in voting: {:f}'.format(end - start))
    f.write('\n')
    f.write('-----------------------------------------')
    f.write('\n\n')
