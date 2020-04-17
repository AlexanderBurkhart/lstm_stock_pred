import pandas as pd
import numpy as np
import datetime as dt
import logging

from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from keras.models import load_model

from optparse import OptionParser

from util import get_data
from model import lstm_model

logging.basicConfig()
log = logging.getLogger('train')
log.setLevel(logging.INFO)

TIME_STEPS = 60
BATCH_SIZE = 20

parser = OptionParser()

parser.add_option('--sd', '--startdate', dest='sd', default='2007-01-01', help='Start date for training.')
parser.add_option('--ed', '--enddate', dest='ed', default='2012-01-01', help='End date for training.')
parser.add_option('--tc', '--traincols', dest='train_cols', 
                    default=['Open','High','Low','Close','Volume'], help='data cols to train on')

(options, args) = parser.parse_args()

sd = dt.datetime.strptime(options.sd, '%Y-%m-%d')
ed = dt.datetime.strptime(options.ed, '%Y-%m-%d')
dates = pd.date_range(sd, ed)

train_cols = options.train_cols
data_sym = 'BAC'
train_size = 0.8
test_size = 0.2

def plot_stock(df):
    plt.figure()
    for col in df_aapl.columns:
        if col == 'Volume':
            continue
        plt.plot(df_aapl[col])

    plt.title('Stock Price History')
    plt.ylabel('Price')
    plt.xlabel('Days')
    plt.legend([col for col in df_aapl.columns])
    plt.show()

def plot_pred(y_pred, y_org):
    plt.figure()
    plt.plot(y_pred)
    plt.plot(y_org)
    plt.title('Prediction vs Real Stock Price')
    plt.ylabel('Price')
    plt.xlabel('Days')
    plt.legend(['Pred', 'Og'], loc='upper left')
    plt.show()

def build_timeseries(mat, y_col_index):
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    print(mat.shape[0])
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0,))
    for i in range(dim_0):
        x[i] = mat[i:TIME_STEPS+i]
        y[i] = mat[TIME_STEPS+i, y_col_index]
    print('length of time-series i/o',x.shape,y.shape)
    return x,y

def trim_dataset(mat, batch_size):
    no_of_rows_drop = mat.shape[0]%batch_size
    if(no_of_rows_drop > 0):
        return mat[:-no_of_rows_drop]
    else:
        return mat

def preprocess(x, y_col_index):
    x,y = build_timeseries(x, y_col_index)
    return trim_dataset(x, BATCH_SIZE), trim_dataset(y, BATCH_SIZE)

log.info('Creating model...')

model = lstm_model((BATCH_SIZE, TIME_STEPS, len(train_cols)))

log.info('Done')
print()

log.info('Loading and preprocessing...')
log.info('Loading data...')

df_data = get_data(data_sym, dates, data_cols=train_cols, sing=True)
df_train, df_test = train_test_split(df_data, train_size=train_size, test_size=test_size, shuffle=False)

log.info('Data loaded.')
print()
log.info('Preprocessing data...')

x = df_train.loc[:,train_cols].values
min_max_scaler = MinMaxScaler()
x_train = min_max_scaler.fit_transform(x)
x_test = min_max_scaler.transform(df_test.loc[:,train_cols])

x_t, y_t = preprocess(x_train, 3)

x_temp, y_temp = preprocess(x_test, 3)
x_val, x_test_t = np.split(x_temp, 2)
y_val, y_test_t = np.split(y_temp, 2)

log.info('Done preprocessing.')
print()
logging.info('Training model...')

history = model.fit(x_t, y_t, epochs=10, verbose=2, batch_size=BATCH_SIZE,
                     shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE),
                     trim_dataset(y_val, BATCH_SIZE)))

#model = load_model('model.h5')

log.info('Done training.')
print()
log.info('Predicting...')

y_pred = model.predict(trim_dataset(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE)
y_pred = y_pred.flatten()
y_test_t = trim_dataset(y_test_t, BATCH_SIZE)

y_pred_org = (y_pred * min_max_scaler.data_range_[3]) + min_max_scaler.data_min_[3]
y_test_t_org = (y_test_t * min_max_scaler.data_range_[3]) + min_max_scaler.data_min_[3]
plot_pred(y_pred_org, y_test_t_org)
