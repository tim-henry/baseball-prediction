#Reorganizing Data files for cumulative data

import numpy as np
import pandas as pd
import os
# BOS and NYA

#---------------------------------------------------

datapath = '../data/GL2004/'
destination = '../data/dest/'
leavout  = ['Name','opp_Name']
#df.drop('b', axis=1)
featureBoundsL = 4
featureBoundsU = 31

#---------------------------------------------------


def cumulative(teamName = 'BOS', gameNumber = 20):
	'''Takes a team name and computes its cumulative stats up to a game number'''
	if gameNumber == 1:
		print('CANNOT GET CUMULATIVE DATA FOR FIRST GAME')
		return(None)
	else:
		try:
			df = pd.read_csv(datapath + teamName + '.csv').iloc[0:gameNumber-1,:]

			colNames = df.columns

			for k in range(0,len(colNames)):
				if (colNames[k] in leavout) or ('opp_' in colNames[k]):
					df = df.drop(colNames[k], axis = 1)

			avs = df.sum(axis = 0).values / (gameNumber)

			colNames = df.columns

			av = pd.DataFrame([avs],columns = colNames)
			#av.iloc[0,:] = a  vs
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

	colNames = df.columns

	tempDF = df
	for k in range(0,len(colNames)):
		if (colNames[k] in leavout) or ('opp_' in colNames[k]):
			tempDF = tempDF.drop(colNames[k], axis = 1)

	names = list(tempDF)
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


def trim_to_cum(df):
	colNames = df.columns.values

	for k,name in enumerate(colNames):
		if (not ('isWin' in name)) and (not('cum' in name) and (not ('isHome' in name))):
			df = df.drop(colNames[k], axis = 1)
	return(df)



def naive(df):
	df = df.iloc[1:,:]
	keep = ['isWin', 'cum_isWin', 'opp_cum_isWin']
	colNames = df.columns.values

	for k,name in enumerate(colNames):
		if (not (name in keep)):
			df = df.drop(colNames[k], axis = 1)

	predictions = []

	for row in range(0,df.shape[0]):
		if (df['cum_isWin'].values[row] >= df['opp_cum_isWin'].values[row]):
			predictions.append(1)
		else:
			predictions.append(0)

	predictions = np.array(predictions)

	actuals = np.array(df['isWin'])

	numWrong = np.sum(np.abs(predictions - actuals))

	numRight = len(predictions) - numWrong

	accuracy = numRight / len(predictions)
	return(accuracy)

def accuracies(source = destination):
	names = os.listdir(source)
	names_filtered = []
	for k in range(0,len(names)):
		if '.csv' in names[k]:
			names_filtered.append(names[k])
	names = names_filtered

	accuracies = []

	for k,name in enumerate(names):
		team = str(name.replace('.csv',''))
		df = pd.read_csv(source + team + '.csv')
		accuracies.append(naive(df))

	return(accuracies)





def main():

	if not os.path.isdir(destination):
		os.makedirs(destination)

	names = os.listdir(datapath)
	names_filtered = []
	for k in range(0,len(names)):
		if '.csv' in names[k]:
			names_filtered.append(names[k])
	names = names_filtered


	for k, name in enumerate(names):
		team = str(name.replace('.csv',''))
		newDF = add_cumulatives(teamName = team)

		newDF.to_csv(destination + team + '.csv')

if __name__ == '__main__':
	#main()
	print('Accuracies: ' + str(np.mean(accuracies())))


