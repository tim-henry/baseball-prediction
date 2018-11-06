#dm.py

import pandas as pd
import numpy as np
import os
import re
import shutil

source = './data'
destination = './temp_working'
colNames = ['Date','GameNumber', 'Day', 'Visitor', 'VisitorLeague','VisitorGameNum','Home','HomeLeague','HomeGameNum','VisitorScore','HomeScore','GameLength','Day/Night','CompletionInf','ForfeitInf','ProtestInf','ParkID','Attendance','TimeOfGame','Vistor_LineScores','Home_LineScores','Visitor_at-bats','Visitor_hits','Visitor_doubles','Visitor_triples','Visitor_Homeruns','Visitor_RBI','Visitor_SacrificeHits','Visitor_SacrificeFlies','Visitor_hit-by-pitch','Visitor_walks','Visitor_intentionalWalks','Visitor_strikeouts','Visitor_StolenBases','Visitor_CaughtStealing','Visitor_GroundedIntoDoublePlays','Visitor_AwardedFirstOnCatcherInterference','Visitor_LeftOnBase','Visitor_PitchersUsed','Visitor_IndividualEarnedRuns','Visitor_TeamEarnedRuns','Visitor_WildPitches','Visitor_Balks','Visitor_putouts','Visitor_assists','Visitor_errors','Visitor_PassedBalls','Visitor_doublePlays','Visitor_TriplePlays','Home_at-bats','Home_hits','Home_doubles','Home_triples','Home_Homeruns','Home_RBI','Home_SacrificeHits','Visitor_SacrificeFlies','Home_hit-by-pitch','Home_walks','Home_intentionalWalks','Home_strikeouts','Home_StolenBases','Home_CaughtStealing','Home_GroundedIntoDoublePlays','Home_AwardedFirstOnCatcherInterference','Home_LeftOnBase','Home_PitchersUsed','Home_IndividualEarnedRuns','Home_TeamEarnedRuns','Home_WildPitches','Home_Balks','Home_putouts','Home_assists','Home_errors','Home_PassedBalls','Home_doublePlays','Home_TriplePlays','HomePlateUmpID','HomePlateUmpName','1BUmpID','1BUmpName','2BUmpID','2BUmpName','3BUmpID','3BUmpName','LFUmpID','LFUmpName','RFUmpID','RFUmpName','Visitor_ManagerID','Visitor_ManagerName','Home_ManagerID','Home_ManagerName','WinningPitcherID','WinningPitcherName','LosingPitcherID','LosingPitcherName','SavingPitcherID','SavingPitcherName','GamewinningRBIBatterID','GamewinningRBIBatterName','Visitor_StartingPitcherID','Visitor_StartingPitcherName','Home_StartingPitcherID','Home_StartingPitcherName','Visitor_Starter1ID','Visitor_Starter1Name','Visitor_Starter1DefensivePosition','Visitor_Starter2ID','Visitor_Starter2Name','Visitor_Starter2DefensivePosition','Visitor_Starter3ID','Visitor_Starter3Name','Visitor_Starter3DefensivePosition','Visitor_Starter4ID','Visitor_Starter4Name','Visitor_Starter4DefensivePosition','Visitor_Starter5ID','Visitor_Starter5Name','Visitor_Starter5DefensivePosition','Visitor_Starter6ID','Visitor_Starter6Name','Visitor_Starter6DefensivePosition','Visitor_Starter7ID','Visitor_Starter7Name','Visitor_Starter7DefensivePosition','Visitor_Starter8ID','Visitor_Starter8Name','Visitor_Starter8DefensivePosition','Visitor_Starter9ID','Visitor_Starter9Name','Visitor_Starter9DefensivePosition','Home_Starter1ID','Home_Starter1Name','Home_Starter1DefensivePosition','Home_Starter2ID','Home_Starter2Name','Home_Starter2DefensivePosition','Home_Starter3ID','Home_Starter3Name','Home_Starter3DefensivePosition','Home_Starter4ID','Home_Starter4Name','Home_Starter4DefensivePosition','Home_Starter5ID','Home_Starter5Name','Home_Starter5DefensivePosition','Home_Starter6ID','Home_Starter6Name','Home_Starter6DefensivePosition','Home_Starter7ID','Home_Starter7Name','Home_Starter7DefensivePosition','Home_Starter8ID','Home_Starter8Name','Home_Starter8DefensivePosition','Home_Starter9ID','Home_Starter9Name','Home_Starter9DefensivePosition','AdditionalInfo','AcquisitionInformation']

teams = ['NYA', 'BOS','TOR']


def nameCols(df):
	''' This is just because i literally never remember how to rename columns'''
	df.columns = colNames
	return(df)

def S_indicator(string, listofstrings, threshhold = 11):
	ind = []
	for idx, s in enumerate(listofstrings):
		if string in s and idx >= threshhold:
			ind.append(True)
		else:
			ind.append(False)

	return(ind)


def availableNames(datapath = source):
	'''SAYS WHICH NAMES ARE AVAILABLE IN THE SPECIFIED FOLDER'''
	data = os.listdir(datapath)
	cleanData = []
	#make sure this contains only csvs
	for name in data:
		if '.TXT' in str(name):
			cleanData.append(str(name))

	data = cleanData
	return(data)



def filterDF(df, by = 'team', ID = 'NYA'):
	idx_visitingTeamName =  3
	idx_homeTeamName = 6
	idx_visitorScore = 9
	idx_homeScore = 10

	df = nameCols(df) #Name the columns according to how that shit is documented



	if by == 'team':
		toKeepList = []
		opponentList = []
		homeList = []

		teamScoreList = []
		opponentScoreList = []

		teamFeatures = []
		opponentFeatures = []

		#nHomeFeatures = sum(S_indicator(string = 'Home'), colNames, threshhold=11)
		#nVisitorFeatures = sum(S_indicator(string = 'Visitor'), colNames, threshhold=11)


		for row in range(0,df.shape[0]):
			if df.iloc[row,idx_visitingTeamName] == ID or df.iloc[row,idx_homeTeamName] == ID:
				toKeepList.append(True)
				if df.iloc[row,idx_visitingTeamName] == ID:
					#The team is playing away
					print('adding away for row ' + str(row))
					opponentList.append(df.iloc[row,idx_homeTeamName])
					homeList.append(0)
					teamScoreList.append(df.iloc[row,idx_visitorScore])
					opponentScoreList.append(df.iloc[row,idx_homeScore])

					#for col in range(11,len(colNames)):

				else:
					#the team is playing at home
					print('adding home for row ' + str(row))
					opponentList.append(df.iloc[row,idx_visitingTeamName])
					homeList.append(1)
					teamScoreList.append(df.iloc[row,idx_homeScore])
					opponentScoreList.append(df.iloc[row,idx_visitorScore])
			else:
				toKeepList.append(False)

		df = df.iloc[toKeepList,]


		df.insert(loc = 1, column = 'Team', value = ID)
		df.insert(loc = 2, column = 'Opponent', value = opponentList)
		df.insert(loc = 3, column = 'AtHome', value = homeList)
		df.insert(loc = 4, column = 'TeamScore', value = teamScoreList)
		df.insert(loc = 5, column = 'OpponentScore', value = opponentScoreList)


		return(df)


def makeRelative(df, threshhold = 11):





def main(source = source, destination = destination):
	if os.path.isdir(destination):
		shutil.rmtree(destination)
	if not os.path.isdir(destination):
		os.mkdir(destination)

	elements = availableNames(source)

	for k in range(0,len(elements)):
		f = elements[k]
		df = pd.read_csv(source + '/' +f)

		year = re.findall('\d+', f)[0]

		os.mkdir(destination + '/' + year)

		for team in teams:
			ndf = filterDF(df, ID = team)

			ndf.to_csv(destination + '/' + year + '/' + team + '.csv')







if __name__ == '__main__':
	main()
