#addLineupsToMA

import pandas as pd
import os

dropbox_dir = os.path.expanduser("~/Documents/Dropbox (MIT)/6.867 - NEW/")

dp_filesWithPlayers = dropbox_dir + 'data_clean_csv_wins_cumulated_withplayers/'

dp_fileswithMA = dropbox_dir + 'data_clean_csv_wins_cumulated_ewm_20/'

destination = dropbox_dir + 'data_clean_csv_wins_cumulated_withplayers_ewm_20/'


dates = list(range(2010,2018))



def main():
	for k, year in enumerate(dates):
		folderName = 'GL' + str(year) + '/'
		files = os.listdir(dp_fileswithMA + folderName)
		files = [s for s in files if '.csv' in s]

		for j, f in enumerate(files):
			ma_df = pd.read_csv(dp_fileswithMA + folderName + f).drop(['Unnamed: 0'], axis = 1)

			lineups = pd.read_csv(dp_filesWithPlayers + folderName + f).iloc[:,125:]

			new_df = pd.concat([ma_df,lineups], axis = 1)

			new_df.to_csv(destination + folderName + f, index = False)


if __name__ == '__main__':
	main()
