import pandas as pd
import matplotlib.pyplot as plt

# Data
dropbox_dir = "/Users/timhenry/Dropbox (MIT)/6.867/"
file = "/GL2017/BOS.csv"

d1 = dropbox_dir + "data_clean_csv_wins_cumulated" + file
d2 = dropbox_dir + "data_clean_csv_wins_cumulated_MA" + file
d3 = dropbox_dir + "data_clean_csv_wins_cumulated_ewm" + file

df1 = pd.read_csv(d1)
df2 = pd.read_csv(d2)
df3 = pd.read_csv(d3)
print(df1.shape)

plt.title("Data Accumulations of BOS Runs over 2017 Season")
plt.ylim(0, 17.5)
plt.xlim(0, 162)
plt.plot(range(df1.shape[0]), df1['Runs'].tolist(), label="raw values", color="blue")
plt.plot(range(df1.shape[0]), df1['cum_Runs'].tolist(), label="season-to-date avg.", color="orange")
plt.plot(range(df1.shape[0]), df2['cum_Runs'].tolist(), label="moving avg.", color="green")
plt.plot(range(df1.shape[0]), df3['cum_Runs'].tolist(), label="exp. weighted mean", color="red")

plt.legend()
plt.savefig("features.png")
