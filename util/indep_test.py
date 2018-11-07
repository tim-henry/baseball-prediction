#Reorganizing Data files for cumulative data

import numpy as np
import pandas as pd
import os
import scipy as sp
from scipy.stats import chi2

# BOS and NYA

#---------------------------------------------------

source = '../../ab6.867/CUM_CONCAT/'
destination = './HYPTEST'
leavout  = ['Name','opp_Name']

#---------------------------------------------------



def diff_space(df):
	'''
	Put a dataframe into diff space
	'''
	Y = df['isWin']

	X = df.drop('isWin',axis = 1)

	nVars = int(X.shape[1]/2)

	XTeam = X.iloc[:,0:nVars]
	XOpp = X.iloc[:,nVars:]

	XDiff = pd.DataFrame(XTeam.values - XOpp.values)
	XDiff.columns = list(XTeam)

	nDF = pd.concat([Y,XDiff], axis = 1)

	return(nDF)



def indep_test(df, var1Col= 'isWin', var2Col = 'cum_isWin', var2Space = [-0.5,0,0.5]):

	numBins = len(var2Space) + 1
	n = df.shape[0]
	#var2Col = df.columns.get_loc(var2Col)

	T = 0
	for k in [0,1]:
		for l in range(0,numBins):
			pkl = df[df[var1Col] == k]
			if l == 0:
				pkl = pkl[pkl[var2Col] <= var2Space[l]]
			elif l < numBins-1:
				pkl = pkl[pkl[var2Col] <= var2Space[l]]
				pkl = pkl[pkl[var2Col] > var2Space[l-1]]
			else:
				pkl = pkl[pkl[var2Col] > var2Space[-1]]
			pkl = pkl.shape[0] / n

			pk = df[df[var1Col] == k]
			pk = pk.shape[0] / n

			pl = df
			if l == 0:
				pl = pl[pl[var2Col] <= var2Space[l]]
			elif l < numBins-1:
				pl = pl[pl[var2Col] <= var2Space[l]]
				pl = pl[pl[var2Col] > var2Space[l-1]]
			else:
				pl = pl[pl[var2Col] > var2Space[-1]]

			pl = pl.shape[0] / n

			#print(k,l)
			#print( pkl, pk, pl)
			#print('Bin:' + str(l))

			try:
				T = T + ((pkl - (pk * pl))**2) / (pk * pl)
			except:
				print('DIV ZERO')

	p = 1 - chi2.cdf(n * T,(2-1) * (numBins - 1))

	return(p)



def main():
	df = pd.read_csv('../../ab6.867/CUM_CONCAT/CUM_CONCAT.csv').drop('isHome', axis = 1).iloc[:,1:]

	df = diff_space(df)

	#testVars = ['cum_isWin', 'cum_Runs', 'cum_RBI', 'cum_Homeruns', 'cum_hits', 'cum_SacrificeHits','cum_IndividualEarnedRuns','cum_TriplePlays','cum_isHome']

	pVals = []
	testVars = list(df.columns.values[1:])

	testVars.remove('cum_GameNum')
	testVars.remove('cum_isHome')
	testVars.remove('cum_AwardedFirstOnCatcherInterference')

	for test in testVars:
		#print('Variable: ' + str(testNum))
		pVals.append(indep_test(df, var1Col = 'isWin', var2Col = test))
		print(test.replace('cum_','')+ ' & ' + str(round(pVals[-1], 5)) + "\\\\")

	pVals = np.array(pVals)

	res = pd.DataFrame(columns = testVars)
	res.loc[0] = pVals 

	print(res)
	print('Corrected Aggregate p-Value:\ ' + str(np.min(res.values) * len(pVals)))

	res.to_csv('./pvals.csv')

	return(res)



if __name__ == '__main__':
	main()

