import pandas as pd
import numpy as np
import datetime as dt
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from util import get_data
from model import lstm_model

TIME_STEPS = 10
BATCH_SIZE = 16

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

sd = dt.datetime(2007,1,1)
ed = dt.datetime(2008,1,1)

dates = pd.date_range(sd, ed)

train_cols = ['Open','High','Low','Close','Volume']

df_aapl = get_data('AAPL', dates, data_cols=train_cols, sing=True)

df_train, df_test = train_test_split(df_aapl, train_size=0.8, test_size=0.2, shuffle=False)
print('Train and Test size', len(df_train), len(df_test))

min_max_scaler = MinMaxScaler()
x_train = min_max_scaler.fit_transform(df_train.loc[:,train_cols])
x_test = min_max_scaler.transform(df_test.loc[:,train_cols])

x_t, y_t = build_timeseries(x_train, 3)
x_t = trim_dataset(x_t, BATCH_SIZE)
y_t = trim_dataset(y_t, BATCH_SIZE)
x_temp, y_temp = build_timeseries(x_test, 3)
x_val, x_test_t = np.split(trim_dataset(x_temp, BATCH_SIZE),2)
y_val, y_test_t = np.split(trim_dataset(y_temp, BATCH_SIZE),2)

model = lstm_model((BATCH_SIZE, TIME_STEPS, x_t.shape[2]))
history = model.fit(x_t, y_t, epochs=10, verbose=2, batch_size=BATCH_SIZE,
                    shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE),
                    trim_dataset(y_val, BATCH_SIZE)))
print('Saving model.')
#model.save('model.h5')
y_pred = model.predict(trim_dataset(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE)
y_pred = y_pred.flatten()
y_test_t = trim_dataset(y_test_t, BATCH_SIZE)

y_pred_org = (y_pred * min_max_scaler.data_range_[3]) + min_max_scaler.data_min_[3]
y_test_t_org = (y_test_t * min_max_scaler.data_range_[3]) + min_max_scaler.data_min_[3]
plot_pred(y_pred_org, y_test_t_org)

#plot_stock(df_aapl)


