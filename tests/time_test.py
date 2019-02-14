import time
import datetime
from centerfinder import util
from centerfinder import sky


def timing(filename: str, bins: int):
    data = util.load_data(filename)
    sk = sky.Sky(data, avg_bins=bins)
    start = time.time()
    sk.vote(108)
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


filename = '../data/SignalN3_mid.txt'
for bins in range(10, 100, 20):
    timing(filename, bins)
