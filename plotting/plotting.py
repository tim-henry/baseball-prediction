#Reorganizing Data files for cumulative data

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
# BOS and NYA

#---------------------------------------------------

destination = './figures/'
mixturePath = '../../playerStats/data_mixtured_pca/'




def dirichletPoints(points = [[-1,1,0,-1],[0,0, np.sqrt(3),0]], color = 'black'):
	plt.plot(points[0],points[1],color = color)



def teamRoster(team = 'BOS', year = year):
	files = os.listdir('../../eventData/' + str(year) + 'eve/')

	files = [s for s in files if '.ROS' in s]

	files = [s for s in files if team in s]

	filename = files[0]

	df = pd.read_csv('../../eventData/' + str(year) + 'eve/' + filename)



	roster = df.iloc[:,0].values

	return(roster)

def playerMixtures(degrees = 3, year = 2013, team = 'BOS', basis = np.array([[-1,0],[1,0],[0,np.sqrt(3)]])):

	data_points = pd.read_csv(mixturePath + str(year) + '/battingMixtures_' + str(degrees) + '.csv')
	roster = teamRoster(team = team, year = year)
	TF = []
	for k in range(0,data_points.shape[0]):
		if data_points['RetroID'][k] in roster:
			TF.append(True)
		else:
			TF.append(False)
	data_points = data_points[TF]

	data_points = data_points[['0','1','2']].values
	memberships = data_points.sum(axis = 0) / data_points.shape[0]
	corner1_pct, corner2_pct, corner3_pct = memberships[0],memberships[1],memberships[2]

	points_transformed = data_points @ basis

	#corner1_pct = np.round(np.sum(data_points[:,0] > 0.9) / len(data_points[:,0]), 2)
	#corner2_pct = np.round(np.sum(data_points[:,1] > 0.9) / len(data_points[:,0]), 2)
	#corner3_pct = np.round(np.sum(data_points[:,2] > 0.9) / len(data_points[:,0]), 2)

	corner_counts = [corner1_pct,corner2_pct,corner3_pct]

	return(points_transformed, corner_counts)





def plot3Mixture(year = 2013, team = 'BOS', color = 'red', show = True, alpha = 0.3):
	#plt.close()

	dirichletPoints(color = 'black')

	pts, corner_counts = playerMixtures(year = year, team = team)

	plt.scatter(pts[:,0],pts[:,1], alpha = alpha, color = color, label = team, s= 200)

	cornerCoords = [[-1,0],[1,0],[0,np.sqrt(3)]]
	offset = [-0.1,-0.1]

	for corner in [0,1,2]:
		plt.annotate(str(np.round(corner_counts[corner] * 100, 2)) + '%', xy=np.array(cornerCoords[corner]) +np.array(offset),bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

	if show:
		plt.show()


#['MIA', 'HOU', 'SLN']
plt.close()
for k, team in enumerate(['NYN', 'HOU', 'SLN','BOS','CLE']):
	plt.close()
	plot3Mixture(team = team, color = ['green','red','blue','orange','purple'][k], show = False, alpha = 0.5, year = 2010)
	plt.legend()
	if team == 'NYN':
		plt.title('NY Mets ' + str(year))
	else:
		plt.title(team + ' ' + str(year))
	#plt.title()
	plt.axis('off')
	#plt.tick_params(
    #	axis='both',          # changes apply to the x-axis
    #	which='both',      # both major and minor ticks are affected
    #	bottom=False,      # ticks along the bottom edge are off
    #	top=False,
    #	left=False,         # ticks along the top edge are off
    #	labelbottom=False)
	#plt.tick_params(
    #	axis='y',          # changes apply to the x-axis
    #	which='both',      # both major and minor ticks are affected
    #	bottom=False,      # ticks along the bottom edge are off
    #	top=False,         # ticks along the top edge are off
    #	labelbottom=False)
	#plt.show()
	plt.savefig('./figures/' + team)


#plt.legend()
#plt.title()
#plt.show()
#plt.savefig('./figures/')


