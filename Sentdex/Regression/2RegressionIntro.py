import pandas as pd 
import math

# import quandl
# df = quandl.get('wiki/GOOGL')
# df.to_csv("data.csv")


df = pd.read_csv('GOOGL.csv')
# print(df)

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# High and Low represents fluctuations in a day however our regression model can't directly detect it
# So we create High Low Percent Change column and daily change
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

# Keeping only the required
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'

# Just in case if missing values, by keeping it as outlier so that it'll be ignored
df.fillna('-99999', inplace=True)

# We want to predict the stock price of next 34th day (i.e, 1% of total days in data, complexity not required, we can hardcode the day)
forecast_out = int(math.ceil(0.01*len(df)))

# create a column label, having the output of next forecast_out's date (i.e, 34th day)
df['label'] = df[forecast_col].shift(-forecast_out)

print(df.head())
df.dropna(inplace=True)
print(df.tail())