#------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib
from matplotlib.dates import DateFormatter, date2num
import matplotlib.pyplot as pp
from vmhmm import VonMisesHMM

import urllib2
import itertools
from datetime import datetime
from cStringIO import StringIO

#------------------------------------------------------------------------
# Dowload data
#------------------------------------------------------------------------

def load_weather(icao, yr=2013, mon=10, day=10):
    "Load the daily weather report from a NWS weather station, by ICAO station name"
    url = 'http://www.wunderground.com/history/airport/%(icao)s/%(yr)04d/%(mon)02d/%(day)02d/DailyHistory.html?format=1' % locals() 
    buf = StringIO(urllib2.urlopen(url).read().replace('<br />', ''))
    return pd.read_csv(buf, header=1, parse_dates=[13], date_parser = lambda x: datetime.strptime(x , '%Y-%m-%d %H:%M:%S'))

# Load data from a few days
dfs = []
for day in range(1, 12):
    print 'Reading day=%d' % day
    # KDSM is the airport code in Des Moines, IA
    df = load_weather('KDSM', day=day)
    df = df[['DateUTC', 'WindDirDegrees']]
    dfs.append(df)
df = pd.concat(dfs)

# Reshape to [n_samples, n_features=1], change to radians
X = df['WindDirDegrees'].values.reshape(-1,1) * (np.pi / 180)

#------------------------------------------------------------------------
# Fit model
#------------------------------------------------------------------------

n_components = 3
model = VonMisesHMM(n_components, n_iter=1000)
model.fit([X])
hidden_states = model.predict(X)

#------------------------------------------------------------------------
# Plot Results
#------------------------------------------------------------------------
pp.figure(figsize=(12,6))
c = itertools.cycle(['r', 'g', 'b'])
for i in range(n_components):
    idx = (hidden_states == i)
    pp.plot_date(date2num(df['DateUTC'][idx]), df['WindDirDegrees'][idx], 'o',
                 c=next(c), label="%dth state" % i, xdate=True)

pp.ylim(-10, 370)
pp.title('von Mises HMM on Wind Direction Data')
pp.ylabel('Wind Direction [deg]')
pp.gcf().autofmt_xdate()
pp.gca().fmt_xdata = DateFormatter('%m-%d')
pp.legend(prop={'size':12})
pp.savefig('winddirection.py.png')
