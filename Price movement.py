import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# read the csv file
data_path = r'C:\Users\28062\Desktop\1\DATA'   # please enter the path of your data file
data_files = sorted(glob.glob(os.path.join(data_path, '**', '*.csv'), recursive=True))


for data_file in data_files:
    df = pd.read_csv(data_file)

    # Convert all time types to Date
    All_types_of_time = {'Date', 'Index', 'Timestamp', 'Datetime', 'Time'}
    date_col = next((t for t in df.columns if t in All_types_of_time), None)
    df.rename(columns={date_col:'Date'},inplace=True)

    # Sort by time
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Price movement
    plt.figure(figsize=(10, 5))
    plt.plot(df['Date'], df['Open'], color='Red', label='Open Price')
    plt.plot(df['Date'], df['Close'], color='blue', label='Close Price')
    plt.title(f'Price Movement ({os.path.basename(data_file)})')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
