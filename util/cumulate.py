#Reorganizing Data files for cumulative data

import numpy as np
import pandas as pd
import os
# BOS and NYA
datapath = '../data/'

featureBoundsL = 4
featureBoundsU = 31


def cumulative(teamName = 'BOS', gameNumber = 20):
	'''Takes a team name and computes its cumulative stats up to a game number'''
	if gameNumber == 1:
		print('CANNOT GET CUMULATIVE DATA FOR FIRST GAME')
		return(None)
	else:
		try:
			df = pd.read_csv(datapath + teamName + '.csv').iloc[0:gameNumber,featureBoundsL:featureBoundsU]

			avs = df.sum(axis = 0).values / gameNumber

			colNames = df.columns

			av = pd.DataFrame([avs],columns = colNames)
			#av.iloc[0,:] = avs
			return(av)
		except:
			return(None)

def addphrase(phrase, listOfStrings):
	newListOfStrings = []
	for idx, s in enumerate(listOfStrings):
		newListOfStrings.append(phrase + s)

	return(newListOfStrings)


def add_cumulatives(teamName = 'BOS'):
	df = pd.read_csv(datapath + teamName + '.csv')

	names = list(df.iloc[:,featureBoundsL:featureBoundsU])
	oppnames = addphrase('opp_cum_',names)
	names = addphrase('cum_', names)

	names.extend(oppnames)

	newDF = pd.DataFrame(index = range(0,df.shape[0]), columns = names)

	for row in range(1,df.shape[0]):
		teamCumulative = cumulative(teamName = teamName, gameNumber = int(df['GameNum'][row]))

		oppCumulative = cumulative(teamName = df['opp_Name'][row], gameNumber = int(df['opp_GameNum'][row]))

		if not((oppCumulative is None) or (teamCumulative is None)):
			cumData = pd.concat([teamCumulative, oppCumulative], axis=1).values
			newDF.iloc[row,:] = cumData

	out = pd.concat([df, newDF], axis=1)

	return(out)

