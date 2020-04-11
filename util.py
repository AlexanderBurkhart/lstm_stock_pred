import pandas as pd
import matplotlib.pyplot as plt
import os

def symbol_to_path(symbol, base_dir='data'):
    return os.path.join(base_dir, '{}.csv'.format(str(symbol)))

def get_data(symbols, dates, index_col='Date', data_cols=['Adj Close'], sing=False):
    #Geting data from only one stock
    if sing:
        file_path = symbol_to_path(symbols)
        df = pd.read_csv(file_path, parse_dates=True, index_col=index_col,
                            usecols=[index_col]+data_cols, na_values=['nan'])
        df = fill_missing_values(df)
        return df

    #Getting data from multiple stocks
    df_data_cols = []
    for data_col in data_cols:
        df_data_col = pd.DataFrame(index=dates)
        for symbol in symbols:
            file_path = symbol_to_path(symbol)
            df_temp = pd.read_csv(file_path, parse_dates=True, index_col=index_col, 
                                    usecols=[index_col, data_col], na_values=['nan'])
            df_temp = df_temp.rename(columns={data_col: symbol})
            df_data_col = df_data_col.join(df_temp)
        df_data_col = fill_missing_values(df_data_col)
        df_data_cols.append(df_data_col)
    df_final = pd.concat(df_data_cols, keys=data_cols)
    return df_final

def plot_data(df_data):
    ax = df_data.plot(title='Stock Data', fontsize=2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    plt.show()

def fill_missing_values(df_data):
    df_data.fillna(method='ffill', inplace=True)
    df_data.fillna(method='bfill', inplace=True)
    return df_data

def calc_stats(df):
    #cumulative return
    crs = []
    for symbol in df.columns:
        data = df[symbol]
        crs.append(data[-1]/data[0] - 1)
    cr = pd.Series(data=crs, index=df.columns)

    #average daily return and standard deviation of daily return
    df_daily = df.copy()
    df_daily[1:] = (df[1:]/df[:-1].values)-1
    df_daily.iloc[0,:] = 0

    adr = df_daily.mean()
    sddr = df_daily.std()

    #sharp ratio
    sr = adr / sddr
    if len(df.columns==1):
        return cr[0], adr[0], sddr[0], sr[0]
    return cr, df_daily, adr, sddr, sr
