#player_featurization
import pandas as pd
import numpy as np
import os

import pybaseball as pb

source = '../../eventData/'



year = 2016

folderName = str(year) + 'eve/'

files = os.listdir()

team = 'ATL'

def eventFileName(year =2016, team = 'ATL', source = source):
	folderName = str(year) + 'eve/'

	files = os.listdir(source + folderName)

	files = [s for s in files if '.EV' in s]

	fileName = [s for s in files if team in s]

	fileName = fileName[0]

	return(fileName)

def eventFile(year = 2016, team = 'ATL', source = source):
	fileName = eventFileName(year = year, team = team, source = source)

	folderName = str(year) + 'eve/'

	df = pd.read_csv(source + folderName + fileName, names = ['Type','Inning','Home','Player','Count','Data','PlayOutcome'])










#df = pd.read_csv(source + folderName + year + team + '.EVE')


