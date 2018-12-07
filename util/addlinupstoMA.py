#addLineupsToMA

import pandas as pd
import os


dp_filesWithPlayers = '../../ab6.867/data_clean_csv_wins_cumulated_withplayers/'

dp_fileswithMA = '../../ab6.867/data_clean_csv_wins_cumulated_MA/'

destination = '../../ab6.867/data_clean_csv_wins_cumulated_withplayers_MA/'


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