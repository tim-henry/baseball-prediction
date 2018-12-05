#Matrix Estimation
import numpy as np
import pandas as pd
import os


datapath = '../../ab6.867/data_raw/'

cumulated_datapath = '../../ab6.867/data_clean_csv_wins_cumulated/'

startYear = 1921

leavout  = ['Name','opp_Name']

#datafile = '../../ab6.867/data_raw/GL2017.TXT'
destination = '../../ab6.867/data_clean_csv_wins_cumulated_withplayers/'


colNames = ['Date', 'GameNum', 'Day', 'Visitor_Name', 'Visitor_League', 'Visitor_GameNum', 'Home_Name', 'Home_League',
            'Home_GameNum', 'Visitor_Runs', 'Home_Runs', 'GameLength', 'Day/Night', 'CompletionInf', 'ForfeitInf',
            'ProtestInf', 'ParkID', 'Attendance', 'TimeOfGame', 'Vistor_LineScores', 'Home_LineScores',
            'Visitor_at-bats', 'Visitor_hits', 'Visitor_doubles', 'Visitor_triples', 'Visitor_Homeruns', 'Visitor_RBI',
            'Visitor_SacrificeHits', 'Visitor_SacrificeFlies', 'Visitor_hit-by-pitch', 'Visitor_walks',
            'Visitor_intentionalWalks', 'Visitor_strikeouts', 'Visitor_StolenBases', 'Visitor_CaughtStealing',
            'Visitor_GroundedIntoDoublePlays', 'Visitor_AwardedFirstOnCatcherInterference', 'Visitor_LeftOnBase',
            'Visiting_PitchersUsed', 'Visiting_IndividualEarnedRuns', 'Visiting_TeamEarnedRuns', 'Visiting_WildPitches',
            'Visiting_Balks', 'Visiting_putouts', 'Visiting_assists', 'Visiting_errors', 'Visiting_PassedBalls',
            'Visiting_doublePlays', 'Visiting_TriplePlays', 'Home_at-bats', 'Home_hits', 'Home_doubles', 'Home_triples',
            'Home_Homeruns', 'Home_RBI', 'Home_SacrificeHits', 'Visitor_SacrificeFlies', 'Home_hit-by-pitch',
            'Home_walks', 'Home_intentionalWalks', 'Home_strikeouts', 'Home_StolenBases', 'Home_CaughtStealing',
            'Home_GroundedIntoDoublePlays', 'Home_AwardedFirstOnCatcherInterference', 'Home_LeftOnBase',
            'Home_PitchersUsed', 'Home_IndividualEarnedRuns', 'Home_TeamEarnedRuns', 'Home_WildPitches', 'Home_Balks',
            'Home_putouts', 'Home_assists', 'Home_errors', 'Home_PassedBalls', 'Home_doublePlays', 'Home_TriplePlays',
            'HomePlateUmpID', 'HomePlateUmpName', '1BUmpID', '1BUmpName', '2BUmpID', '2BUmpName', '3BUmpID',
            '3BUmpName', 'LFUmpID', 'LFUmpName', 'RFUmpID', 'RFUmpName', 'Visiting_ManagerID', 'Visiting_ManagerName',
            'Home_ManagerID', 'Home_ManagerName', 'WinningPitcherID', 'WinningPitcherName', 'LosingPitcherID',
            'LosingPitcherName', 'SavingPitcherID', 'SavingPitcherName', 'GamewinningRBIBatterID',
            'GamewinningRBIBatterName', 'Visiting_StartingPitcherID', 'Visiting_StartingPitcherName',
            'Home_StartingPitcherID', 'Home_StartingPitcherName', 'Visiting_Starter1ID', 'Visiting_Starter1Name',
            'Visiting_Starter1DefensivePosition', 'Visiting_Starter2ID', 'Visiting_Starter2Name',
            'Visiting_Starter2DefensivePosition', 'Visiting_Starter3ID', 'Visiting_Starter3Name',
            'Visiting_Starter3DefensivePosition', 'Visiting_Starter4ID', 'Visiting_Starter4Name',
            'Visiting_Starter4DefensivePosition', 'Visiting_Starter5ID', 'Visiting_Starter5Name',
            'Visiting_Starter5DefensivePosition', 'Visiting_Starter6ID', 'Visiting_Starter6Name',
            'Visiting_Starter6DefensivePosition', 'Visiting_Starter7ID', 'Visiting_Starter7Name',
            'Visiting_Starter7DefensivePosition', 'Visiting_Starter8ID', 'Visiting_Starter8Name',
            'Visiting_Starter8DefensivePosition', 'Visiting_Starter9ID', 'Visiting_Starter9Name',
            'Visiting_Starter9DefensivePosition', 'Home_Starter1ID', 'Home_Starter1Name',
            'Home_Starter1DefensivePosition', 'Home_Starter2ID', 'Home_Starter2Name', 'Home_Starter2DefensivePosition',
            'Home_Starter3ID', 'Home_Starter3Name', 'Home_Starter3DefensivePosition', 'Home_Starter4ID',
            'Home_Starter4Name', 'Home_Starter4DefensivePosition', 'Home_Starter5ID', 'Home_Starter5Name',
            'Home_Starter5DefensivePosition', 'Home_Starter6ID', 'Home_Starter6Name', 'Home_Starter6DefensivePosition',
            'Home_Starter7ID', 'Home_Starter7Name', 'Home_Starter7DefensivePosition', 'Home_Starter8ID',
            'Home_Starter8Name', 'Home_Starter8DefensivePosition', 'Home_Starter9ID', 'Home_Starter9Name',
            'Home_Starter9DefensivePosition', 'AdditionalInfo', 'AcquisitionInformation']


'''
df = pd.read_csv(datafile, header = None)

df.columns = colNames


teamList = df['Home_Name'].unique()

dm = np.zeros((len(teamList),len(teamList)))

filter_col = [col for col in df if col.startswith('Home_Starter')]
filter_col = [col for col in df[filter_col] if col.endswith('ID')]
players = df[filter_col]

playerList = np.array([])
for col in players.columns:
	playerList = np.append(playerList,players[col].values)

playerList = np.unique(playerList)'''


def availableYears(datapath = datapath):
	files = os.listdir(datapath)



def seasonDF(year = 2017):
	datafile = datapath + 'GL' + str(year) + '.TXT'
	df = pd.read_csv(datafile, header = None)
	df.columns = colNames
	return(df)



def seasonPlayers(seasonDF):
	df = seasonDF
	df.columns = colNames
	filter_col = [col for col in df if col.startswith('Home_Starter')]
	filter_col = [col for col in df[filter_col] if col.endswith('ID')]
	players = df[filter_col]
	playerList = np.array([])
	for col in players.columns:
		playerList = np.append(playerList,players[col].values)

	#print(playerList)
	playerList = np.unique(playerList.astype(str))

	return(playerList)


def allPlayers(years = range(1921,2018)):
	players = np.array([])

	for year in years:
		#print(year)
		splayers = seasonPlayers(seasonDF(year = year))

		players = np.append(players, splayers)

	players = np.unique(players)

	return(players)

def getRoster(teamName = 'BOS', year = 2017, gameNumber = 20):
	df = seasonDF(year = year)
	df = df[(df['Home_Name'] == teamName) + (df['Visitor_Name'] == teamName)]
	#print(df)
	#print(gameNumber)
	gameDF = df.iloc[int(gameNumber - 1),:]

	atHome = (gameDF['Home_Name'] == teamName)


	if atHome:
		#print('filtering Cls')
		filter_col = [col for col in df if col.startswith('Home_Starter')]
		filter_col = [col for col in df[filter_col] if col.endswith('ID')]
		df = gameDF[filter_col]
		players = df.values
		#print('players:' + str(players))
		return(players)
	else:
		filter_col = [col for col in df if col.startswith('Visiting_Starter')]
		filter_col = [col for col in df[filter_col] if col.endswith('ID')]
		df = gameDF[filter_col]
		players = df.values
		return(players)


def StarterFeature(playerDict,teamName = 'BOS', year = 2017, gameNumber=20):
	players =playerDict

	roster = getRoster(teamName = teamName, year = year, gameNumber = gameNumber)

	vec = np.zeros(len(players))

	for k in range(0,len(players)):
		if players[k] in roster:
			vec[k] = 1

	return(vec)


def addStarters(datapath = cumulated_datapath, target = destination, years = range(1921,2018) ):
	playerDict = list(allPlayers(years = years))
	opp_playerDict = ['opp_' + s for s in playerDict]

	for year in years:
		filepath = datapath + 'GL' + str(year)
		files = os.listdir(filepath)
		print(year)

		for f in files:
			if f.endswith('.csv'):
				df = pd.read_csv(filepath + '/' + f)
				name = df['Name'][0]
				print(name)


				newDF = newDF = pd.DataFrame(index = range(0,df.shape[0]), columns = playerDict + opp_playerDict)

				for row in range(0,df.shape[0]):
					feat = StarterFeature(playerDict = playerDict, teamName = name, year = year, gameNumber = row + 1)
					opp_GameNum = df['opp_GameNum'][row]
					opp_teamName = df['opp_Name'][row]
					#print(opp_GameNum)
					#print(opp_teamName)
					opp_feat = StarterFeature(playerDict = playerDict, teamName = opp_teamName, year = year, gameNumber = opp_GameNum)
					newDF.iloc[row,:] = np.append(feat, opp_feat)

				out = pd.concat([df, newDF], axis=1)

				out.to_csv('../../ab6.867/data_clean_csv_wins_cumulated_withplayers/GL' + str(year) + '/' + name + '.csv', index = False)









def main():

	addStarters(years = range(2010,2018))


if __name__ == '__main__':
	main()













