#plt
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt




def main():
	t = []
	for year in list(range(2010,2018)):
		source = '../../ab6.867/data_clean_csv_wins_cumulated/GL' + str(year) + '/'

		l = os.listdir(source)

		l = [s for s in l if '.csv' in s]

		print(year)

		for k, team in enumerate(l):
			#print(team)
			wins = pd.read_csv(source + team)['isWin'].values
			wins = (wins *2) - np.ones(len(wins))
			s = [0]
			for j,game in enumerate(wins):
				s.append(np.sum(wins[0:j]))
				plt.plot(s)

			t.append(s[-1])

	plt.show()

	plt.close()
	plt.hist(t, bins = 20)
	plt.show()



if __name__ == '__main__':
	main()