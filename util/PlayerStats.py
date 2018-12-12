import pybaseball as pb
import pandas as pd
import numpy as np
import os




eventFilePath = '../../eventData/'

playerDatapath = '../../playerStats/data_raw/'




for row in range(3980,df.shape[0]):

	#df = pd.read_csv('./player_lookup/players.csv')

	ID = df['RetroID'][row]
	relevant = df[df['RetroID'] == ID]
	#print(relevant.shape)
	LastName = relevant['LastName'].values[0]
	FirstName = relevant['FirstName'].values[0]
	ndf.iloc[row,1] = LastName
	ndf.iloc[row,2] = FirstName

	playerInfo = pb.playerid_lookup(last = LastName, first = FirstName)#['key_mlbam']
	playerInfo = playerInfo[playerInfo['key_retro'] == ID]
	playerInfo = playerInfo['key_mlbam']
	df.iloc[row,3] = playerInfo.values
	print(row)


	if row % 100 == 0:
		print('row ' + str(row) + ' of ' + str(ndf.shape[0]))
		#ndf.to_csv('player_lookup/players.csv', index = False)

ndf.to_csv('player_lookup/players.csv', index = False)





for row in range(0,playerdf.shape[0]):
	try:
		playerdf['MLBAM'][row] = int(playerdf['MLBAM'][row].replace(']','').replace('[',''))
	except:
		print('Error at row ' + str(row))
		print('value is ')
		print(playerdf['MLBAM'][row])




'''
ADDING THE IDENTIFIERS TO THE BATTING STATISTICS
'''


playerdf = pd.read_csv('./player_lookup/players.csv')


for year in list(range(2011,2019)):
	print('*********   YEAR:  ' + str(year))
	ambiguousCount = 0
	teamCodes = pd.read_csv('teamCodes.csv')
	battingdf = pd.read_csv('./data_byYear/' + str(year) + '/battingStats.csv').drop('RetroID', axis = 1).drop('MLBAM', axis = 1)
	battingdf.insert(loc = 1, column = 'MLBAM', value = '')
	battingdf.insert(loc = 1,column = 'RetroID', value = '')
	for row in range(0,battingdf.shape[0]):
		#print(row)
		if row % 100==0:
			print('on row ' + str(row))
		name = battingdf['Name'][row]
		name = name.split(' ',1)
		FirstName = name[0]
		LastName = name[1]
		team = battingdf['Team'][row]

		relevant = playerdf[playerdf['LastName'] == LastName]
		relevant = relevant[relevant['FirstName'] == FirstName]

		if LastName == 'Reyes' and FirstName == 'Jose':
			print('JOSE')
			print (relevant)

		if relevant.shape[0] > 1:
			#Player is Ambiguous, manually search for ids using team name
			#relevant = pb.playerid_lookup(last = LastName, first = FirstName)
			#relevant = relevant[relevant]
			print('RESOLVING AMBIGUITIES')
			ambiguousCount = ambiguousCount + 1

			teamCode = teamCodes[teamCodes['NICKNAME'] == team]

			if LastName == 'Reyes' and FirstName == 'Jose':
				print('JOSE')
				print (teamCode)

			if teamCode.shape[0] > 0:
				teamCode = teamCode['TEAMID'].values[-1]
				print(teamCode)

				eventFolder = '../eventData/' + str(year) + 'eve/'
				files = os.listdir(eventFolder)
				files = [s for s in files if '.ROS' in s]
				files = [s for s in files if teamCode in s]
				if len(files) >0:
					roster = pd.read_csv(eventFolder + files[0],header = None)
					possible_RetroIDs = relevant['RetroID'].values
					if LastName == 'Reyes' and FirstName == 'Jose':
						print('POSSIBLE RETROS')
						print(possible_RetroIDs)
						print(roster)

				#possible_MLBAMIDs = relevant['MLBAM'].values
					for k, RetroID in enumerate(possible_RetroIDs):
						#print('LOOKING')
						if RetroID in roster.iloc[:,0].values:
							#print('FOUND IT')
							#try:
							battingdf['RetroID'][row] = possible_RetroIDs[k]
						#except:
						#	print('... Error resolving ambiguity')
								#try:
							#	MLBAM = possible_MLBAMIDs[k]
							#	battingdf['MLBAM'][row] = MLBAM
							#except:
							#	print('... Error resolving ambiguity')
			#except:
			#	'ERROR in teamcode'
		else:
			try:
				RetroID = relevant['RetroID'].values[0]
			except:
				RetroID = relevant['RetroID'].values
			battingdf['RetroID'][row] = RetroID
			#try:
			#	#MLBAM = int(relevant['MLBAM'].values[0])
			#	battingdf['RetroID'][row] = RetroID
			#	#battingdf['MLBAM'][row] = MLBAM
			#except:
			#	print('error at row ' + str(row))
			#	print('FirstName ' + str(FirstName) + '   LastName: ' + str(LastName))
			#	print('RetroID: ' + str(RetroID))
				#print('value is')
				#print(relevant['RetroID'])
			#try:
			#	MLBAM = int(relevant['MLBAM'].values[0])
			#except:
			#	MLBAM = int(relevant['MLBAM'].values)
			#battingdf['MLBAM'][row] = MLBAM
			#try:
				#RetroID = relevant['RetroID'].values[0]
				#battingdf['RetroID'][row] = RetroID
			#	battingdf['MLBAM'][row] = MLBAM
			#except:
			#	print('error at row ' + str(row))
			#	print('FirstName ' + str(FirstName) + '   LastName: ' + str(LastName))
			#	print('RetroID: ' + str(MLBAM))
				#print('value is')
				#print(relevant['RetroID'])
	print(str(ambiguousCount) + ' ambiguities resolved')
	battingdf.to_csv('./data_byYear/' + str(year) + '/battingStats.csv', index=False)

'''
ADD IDENTIFERS TO PITCHERS AGAIN
'''


playerdf = pd.read_csv('./player_lookup/players.csv')


for year in list(range(2008,2019)):
	print('*********   YEAR:  ' + str(year))
	ambiguousCount = 0
	teamCodes = pd.read_csv('teamCodes.csv')
	pitchingdf = pd.read_csv('./data_byYear/' + str(year) + '/pitchingStats.csv').drop('RetroID', axis = 1).drop('MLBAM', axis = 1)
	pitchingdf.insert(loc = 1, column = 'MLBAM', value = '')
	pitchingdf.insert(loc = 1,column = 'RetroID', value = '')
	for row in range(0,pitchingdf.shape[0]):
		#print(row)
		if row % 100==0:
			print('on row ' + str(row))
		name = pitchingdf['Name'][row]
		name = name.split(' ',1)
		FirstName = name[0]
		LastName = name[1]
		team = pitchingdf['Team'][row]

		relevant = playerdf[playerdf['LastName'] == LastName]
		relevant = relevant[relevant['FirstName'] == FirstName]

		if LastName == 'Reyes' and FirstName == 'Jose':
			print('JOSE')
			print (relevant)

		if relevant.shape[0] > 1:
			#Player is Ambiguous, manually search for ids using team name
			#relevant = pb.playerid_lookup(last = LastName, first = FirstName)
			#relevant = relevant[relevant]
			print('RESOLVING AMBIGUITIES')
			ambiguousCount = ambiguousCount + 1

			teamCode = teamCodes[teamCodes['NICKNAME'] == team]

			if LastName == 'Reyes' and FirstName == 'Jose':
				print('JOSE')
				print (teamCode)

			if teamCode.shape[0] > 0:
				teamCode = teamCode['TEAMID'].values[-1]
				print(teamCode)

				eventFolder = '../eventData/' + str(year) + 'eve/'
				files = os.listdir(eventFolder)
				files = [s for s in files if '.ROS' in s]
				files = [s for s in files if teamCode in s]
				if len(files) >0:
					roster = pd.read_csv(eventFolder + files[0],header = None)
					possible_RetroIDs = relevant['RetroID'].values
					if LastName == 'Reyes' and FirstName == 'Jose':
						print('POSSIBLE RETROS')
						print(possible_RetroIDs)
						print(roster)

				#possible_MLBAMIDs = relevant['MLBAM'].values
					for k, RetroID in enumerate(possible_RetroIDs):
						#print('LOOKING')
						if RetroID in roster.iloc[:,0].values:
							#print('FOUND IT')
							#try:
							pitchingdf['RetroID'][row] = possible_RetroIDs[k]
						#except:
						#	print('... Error resolving ambiguity')
								#try:
							#	MLBAM = possible_MLBAMIDs[k]
							#	battingdf['MLBAM'][row] = MLBAM
							#except:
							#	print('... Error resolving ambiguity')
			#except:
			#	'ERROR in teamcode'
		else:
			try:
				RetroID = relevant['RetroID'].values[0]
			except:
				RetroID = relevant['RetroID'].values
			pitchingdf['RetroID'][row] = RetroID
			#try:
			#	#MLBAM = int(relevant['MLBAM'].values[0])
			#	battingdf['RetroID'][row] = RetroID
			#	#battingdf['MLBAM'][row] = MLBAM
			#except:
			#	print('error at row ' + str(row))
			#	print('FirstName ' + str(FirstName) + '   LastName: ' + str(LastName))
			#	print('RetroID: ' + str(RetroID))
				#print('value is')
				#print(relevant['RetroID'])
			#try:
			#	MLBAM = int(relevant['MLBAM'].values[0])
			#except:
			#	MLBAM = int(relevant['MLBAM'].values)
			#battingdf['MLBAM'][row] = MLBAM
			#try:
				#RetroID = relevant['RetroID'].values[0]
				#battingdf['RetroID'][row] = RetroID
			#	battingdf['MLBAM'][row] = MLBAM
			#except:
			#	print('error at row ' + str(row))
			#	print('FirstName ' + str(FirstName) + '   LastName: ' + str(LastName))
			#	print('RetroID: ' + str(MLBAM))
				#print('value is')
				#print(relevant['RetroID'])
	print(str(ambiguousCount) + ' ambiguities resolved')
	pitchingdf.to_csv('./data_byYear/' + str(year) + '/pitchingStats.csv', index=False)





'''
ADD IDENTIFERS TO PITCHERS
'''


for year in list(range(2008,2019)):
	print('*********   YEAR:  ' + str(year))
	ambiguousCount = 0
	teamCodes = pd.read_csv('teamCodes.csv')
	pitchingdf = pd.read_csv('./data_byYear/' + str(year) + '/pitchingStats.csv').drop('RetroID', axis = 1).drop('MLBAM', axis = 1)
	pitchingdf.insert(loc = 1, column = 'MLBAM', value = '')
	pitchingdf.insert(loc = 1,column = 'RetroID', value = '')
	for row in range(0,pitchingdf.shape[0]):
		if row % 100==0:
			print('on row ' + str(row))
		name = pitchingdf['Name'][row]
		name = name.split(' ',1)
		FirstName = name[0]
		LastName = name[1]
		team = pitchingdf['Team'][row]

		relevant = playerdf[playerdf['LastName'] == LastName]
		relevant = relevant[relevant['FirstName'] == FirstName]

		if relevant.shape[0] > 1:
			#Player is Ambiguous, manually search for ids using team name
			#relevant = pb.playerid_lookup(last = LastName, first = FirstName)
			#relevant = relevant[relevant]
			print('RESOLVING AMBIGUITIES')
			ambiguousCount = ambiguousCount + 1

			teamCode = teamCodes[teamCodes['NICKNAME'] == team]

			try:
				teamCode = teamCode['TEAMID'].values[-1]

				eventFolder = '../eventData/' + str(year) + 'eve/'
				files = os.listdir(eventFolder)
				files = [s for s in files if '.ROS' in s]
				files = [s for s in files if teamCode in s]
				roster = pd.read_csv(eventFolder + files[0])
				possible_RetroIDs = relevant['RetroID'].values
				possible_MLBAMIDs = relevant['MLBAM'].values
				for k, RetroID in enumerate(possible_RetroIDs):
					if RetroID in roster:
						try:
							RetroID = possible_RetroIDs[k]
							pitchingdf['RetroID'][row] = RetroID
						except:
							print('... Error resolving ambiguity')
						try:
							MLBAM = possible_MLBAMIDs[k]
							pitchingdf['MLBAM'][row] = MLBAM
						except:
							print('... Error resolving ambiguity')
			except:
				'ERROR in teamcode'
		else:
			try:
				RetroID = relevant['RetroID'].values[0]
				#MLBAM = int(relevant['MLBAM'].values[0])
				pitchingdf['RetroID'][row] = RetroID
				#battingdf['MLBAM'][row] = MLBAM
			except:
				print('error at row ' + str(row))
				#print('value is')
				#print(relevant['RetroID'])

			try:
				#RetroID = relevant['RetroID'].values[0]
				MLBAM = int(relevant['MLBAM'].values[0])
				#battingdf['RetroID'][row] = RetroID
				pitchingdf['MLBAM'][row] = MLBAM
			except:
				print('error at row ' + str(row))
				#print('value is')
				#print(relevant['RetroID'])
	print(str(ambiguousCount) + ' ambiguities resolved')
	pitchingdf.to_csv('./data_byYear/' + str(year) + '/pitchingStats.csv', index=False)


'''
**********************************************************
'''



for year in list(range(2008,2019)):
	ndf = pd.read_csv('./data_byYear/' + str(year) + '/pitchingStats.csv')

	if year == 2008:
		df = ndf
	else:
		df = pd.concat([df,ndf])





for year in list(range(2008,2019)):
	ndf = pd.read_csv('./data_byYear/' + str(year) + '/battingStats.csv')

	if year == 2008:
		df = ndf
	else:
		df = pd.concat([df,ndf])







