#Mixtures
import pandas as pd
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import theano.tensor as tt
import sklearn.mixture as mixture
from sklearn.decomposition import PCA
import os

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

#cluster_features = ['G','AB','PA','H','1B','2B','3B','HR','R','RBI','BB','IBB','SO','HBP','SF','SH','GDP','SB','CS','AVG','GB','FB','LD','IFFB','Pitches','Balls','Strikes']
#cluster_features = ['1B','2B','3B','HR','R','RBI','SB','CS']
cumulated_datapath = '../../ab6.867/data_clean_csv_wins_cumulated/'
destination = '../../ab6.867/data_clean_csv_wins_cumulated_withplayers_transformed/'
#cluster_features_batting = ['SB','1B','2B','3B','BB%','K%','BB/K','OBP','GB/FB','BABIP','GB%','LD%','FB%','HR/FB']
cluster_features_batting = ['SB','BB%','K%','BB/K','OBP','GB/FB','BABIP','GB%','LD%','FB%','HR/FB','Pull%','Cent%','Oppo%','Soft%','Med%','Hard%']


features_pitching = ['ERA','K/9','BB/9','HR/9','xFIP','FA% (pfx)','CU% (pfx)']


battingStats = '../../playerStats/starter_battingStats.csv'
pitchingStats = '../../playerStats/starter_pitchingStats.csv'
mixtureTargets = '../../playerStats/data_mixtured_pca/'

player_lookup_file = '../../playerStats/player_lookup/'


datapath = '../../ab6.867/data_raw/'
leavout  = ['Name','opp_Name']


def battingData():
	df = pd.read_csv(battingStats)[cluster_features_batting]
	return(df)

def pitchingData():
	df = pd.read_csv(pitchingStats)[features_pitching]
	return(df)


def likelihoodCurve(df, kRange = list(range(1,10))):
	likelihoods = np.array([])
	for k in kRange:
		print('N Clusters ' + str(k))
		mod = mixture.BayesianGaussianMixture(n_components = k, max_iter = 100, n_init = 1)
		mod.fit(df.values)
		likelihood = mod.score(df.values)
		if k== kRange[0]:
			likelihoods = np.array([k, likelihood])
		else:
			likelihoods = np.vstack([likelihoods,np.array([k, likelihood])])

	return(np.array(likelihoods))

#def RetroID_to_name(RetroID = 'fernj003'):



def creatMixtureValues(years = list(range(2008,2019)), n_components = 3):
	battingdf = battingData()

	#create the model
	mod = mixture.BayesianGaussianMixture(n_components = n_components, max_iter = 300, n_init = 4)
	mod.fit(battingdf.dropna().values)

	#Get the responsibilities
	IDS = pd.read_csv(battingStats)[['Season','RetroID','MLBAM']]

	memberships = mod.predict_proba(battingdf.fillna(battingdf.mean()).values)

	memberships = pd.DataFrame(memberships, columns = range(0,n_components))

	out = pd.concat([IDS, memberships], axis = 1)


	for year in years:
		outdf = out[out['Season'] == year]
		outdf.to_csv(mixtureTargets + str(year) + '/battingMixtures_' + str(n_components) + '.csv', index = False)


def createPCAValues(years = list(range(2008,2019)), n_components = 3):
	battingdf = battingData()

	mod = PCA(n_components = n_components)

	mod.fit(battingdf.dropna().values)

	IDS = pd.read_csv(battingStats)[['Season','RetroID','MLBAM']]

	loadings = mod.transform(battingdf.fillna(battingdf.mean()).values)

	loadings = pd.DataFrame(loadings, columns = range(0,n_components))

	out = pd.concat([IDS, loadings], axis = 1)

	for year in years:
		outdf = out[out['Season'] == year]
		outdf.to_csv(mixtureTargets + str(year) + '/battingPCA_' + str(n_components) + '.csv', index = False)




def createPitchingPCAValues(years = list(range(2008,2019)), n_components = 3):
	pitchingdf = pitchingData()

	mod = PCA(n_components = n_components)

	mod.fit(pitchingdf.dropna().values)

	IDs = pd.read_csv(pitchingStats)[['Season','RetroID','MLBAM']]

	loadings = mod.transform(pitchingdf.fillna(pitchingdf.mean()).values)

	loadings = pd.DataFrame(loadings, columns = range(0,n_components))

	out = pd.concat([IDs,loadings], axis = 1)

	for year in years:
		outdf = out[out['Season'] == year]
		outdf.to_csv(mixtureTargets + str(year) + '/pitchingPCA_' + str(n_components) + '.csv', index = False)





def seasonDF(year = 2017):
	datafile = '../../ab6.867/data_raw/' + 'GL' + str(year) + '.TXT'
	df = pd.read_csv(datafile, header = None)
	df.columns = colNames
	return(df)


def getRoster(teamName = 'MIA', year = 2017, gameNumber = 20):
	df = seasonDF(year = year)
	df = df[(df['Home_Name'] == teamName) | (df['Visitor_Name'] == teamName)]
	#print(df)
	#print(gameNumber)
	if df.shape[0] >= 1 and gameNumber <= df.shape[0]:
		gameDF = df.iloc[int(gameNumber - 1),:]
		atHome = (gameDF['Home_Name'] == teamName)
		if atHome:
			#print('filtering Cls')
			filter_col = [col for col in df if col.startswith('Home_Starter')]
			filter_col = [col for col in df[filter_col] if col.endswith('ID')]

			#positions_col = [col for col in df if col.startswith('Home_Starter')]
			#positions_col = [col for col in df[positions_col] if col.endswith('DefensivePosition')]
			#positions = gameDF[positions_col].values
			#try:
			#	pitcher_idx = np.where(positions == 1)[0][0]
			#except:
			#	pitcher_idx = np.where(positions == 10)[0][0]
			df = gameDF[filter_col]
			batters = df.values
			#print('players:' + str(players))

			pitcher = gameDF['Home_StartingPitcherID']


			return(pitcher, batters)
		else:
			filter_col = [col for col in df if col.startswith('Visiting_Starter')]
			filter_col = [col for col in df[filter_col] if col.endswith('ID')]

			#positions_col = [col for col in df if col.startswith('Visiting_Starter')]
			#positions_col = [col for col in df[positions_col] if col.endswith('DefensivePosition')]
			#positions = gameDF[positions_col].values
			#try:
			#	pitcher_idx = np.where(positions == 1)[0][0]
			#except:
			#	pitcher_idx = np.where(positions == 10)[0][0]
			df = gameDF[filter_col]
			batters = df.values

			pitcher = gameDF['Visiting_StartingPitcherID']
			#batters = np.delete(batters, pitcher_idx)
			#print('players:' + str(players))
			return(pitcher, batters)
	else:
		print('TEAM GAMES NOT FOUND:' + teamName + ' Game: ' + str(gameNumber))
		return(None,[None]*8 )


def pitcher_features(RetroID= 'thomj007', year = 2017, n_components=3):
	pitchingdf = pd.read_csv(mixtureTargets + str(year) + '/pitchingPCA_' + str(n_components) + '.csv')

	stats = pitchingdf[pitchingdf['RetroID'] == RetroID].drop(['Season','RetroID','MLBAM'], axis = 1)

	if stats.shape[0] ==0:
		print('Pitcher Features ('+ str(RetroID) +') Not Found in Year ' +str(year) + '. Using Season Average as Default.')
		#print('retro:' + str(RetroID))
		#print(stats)

		stats = pitchingdf.drop(['Season','RetroID','MLBAM'], axis = 1).mean(axis = 0)
	else:
		#print('pitcher found')
		stats = stats.iloc[0,:]
	return(stats.values)

def batters_features_mixture(RetroIDs =['benia002', 'bettm001', 'morem001', 'ramih003', 'bradj001','rutlj001', 'hernm003', 'leons001'], year = 2017, n_components = 3):
	battingdf = pd.read_csv(mixtureTargets + str(year) + '/battingMixtures_' + str(n_components) + '.csv')
	stats = battingdf[battingdf['RetroID'].isin(RetroIDs)].drop(['Season','RetroID','MLBAM'], axis =1)

	if stats.shape[0]==0:
		print('No Batter Info Found. Using Season Average as Default')
		stats = battingdf.drop(['Season','RetroID','MLBAM'], axis = 1).mean(axis =0)
	else:
		stats = stats.mean(axis = 0)
	return(stats.values)

def batters_features_pca(RetroIDs =['benia002', 'bettm001', 'morem001', 'ramih003', 'bradj001','rutlj001', 'hernm003', 'leons001'], year = 2017, n_components = 3):
	battingdf = pd.read_csv(mixtureTargets + str(year) + '/battingPCA_' + str(n_components) + '.csv')
	stats = battingdf[battingdf['RetroID'].isin(RetroIDs)].drop(['Season','RetroID','MLBAM'], axis =1)

	if stats.shape[0]==0:
		print('No Batter Info Found. Using Season Average as Default')
		stats = battingdf.drop(['Season','RetroID','MLBAM'], axis = 1).mean(axis =0)
	else:
		stats = stats.mean(axis = 0)
	return(stats.values)


def add_player_info(datapath = cumulated_datapath, target = destination, years = range(2010,2018), num_pitcher_features = 5, num_batter_features = 3):
	for year in years:
		filepath = datapath + 'GL' + str(year)
		files = os.listdir(filepath)

		files = [s for s in files if '.csv' in s]

		for f in files:
			df = pd.read_csv(filepath + '/' + f).drop(['Unnamed: 0'], axis = 1)
			name = df['Name'][0]
			print(name)

			pitcher_feats = list(range(0,num_pitcher_features))
			pitcher_feats = ['pitcher_' + str(s) for s in pitcher_feats]

			batter_feats = list(range(0,num_batter_features))
			batter_feats = ['batter_' + str(s) for s in batter_feats]

			teamFeats = pitcher_feats + batter_feats

			#teamFeats = list(np.append(np.array(range(0,num_pitcher_features)),np.array(range(0,num_batter_features))))
			


			oppFeats = ['opp_' + str(s) for s in teamFeats]
			#teamFeats = [s for s in teamFeats]

			newDF_mixt = pd.DataFrame(index = range(0,df.shape[0]), columns = teamFeats + oppFeats)
			newDF_pca = pd.DataFrame(index = range(0,df.shape[0]), columns = teamFeats + oppFeats)


			for row in range(0,df.shape[0]):
				pitcher , batters = getRoster(teamName = name, year = year - 1, gameNumber = row+1)

				pitcherStats = pitcher_features(pitcher, year = year- 1, n_components = num_pitcher_features)
				batterStats_mixt = batters_features_mixture(RetroIDs = list(batters), year = year- 1, n_components = num_batter_features)
				batterStats_pca = batters_features_pca(RetroIDs = list(batters), year = year- 1, n_components = num_batter_features)

				feat_mixt = np.array(list(pitcherStats) + list(batterStats_mixt))
				feat_pca = np.array(list(pitcherStats) + list(batterStats_pca))

				opp_GameNum = df['opp_GameNum'][row]
				opp_teamName = df['opp_Name'][row]
				pitcher , batters = getRoster(teamName = opp_teamName, year = year- 1, gameNumber = opp_GameNum)

				pitcherStats = pitcher_features(pitcher, year = year- 1, n_components = num_pitcher_features)
				
				batterStats_mixt = batters_features_mixture(RetroIDs = list(batters), year = year- 1, n_components = num_batter_features)
				batterStats_pca = batters_features_pca(RetroIDs = list(batters), year = year- 1, n_components = num_batter_features)

				opp_feat_mixt = np.array(list(pitcherStats) + list(batterStats_mixt))
				opp_feat_pca = np.array(list(pitcherStats) + list(batterStats_pca))

				newDF_mixt.iloc[row,:] = np.append(feat_mixt, opp_feat_mixt)
				newDF_pca.iloc[row,:] = np.append(feat_pca, opp_feat_pca)

			out_mixt = pd.concat([df, newDF_mixt], axis = 1)
			out_pca = pd.concat([df,newDF_pca], axis = 1)

			out_mixt.to_csv('../../ab6.867/data_clean_csv_wins_cumulated_withplayers_transformed/' + 'mixture' + '/' + 'num_degrees' + str(num_batter_features) + '/GL' + str(year) + '/' + name + '.csv', index = False)
			out_pca.to_csv('../../ab6.867/data_clean_csv_wins_cumulated_withplayers_transformed/' + 'pca' + '/' + 'num_degrees' + str(num_batter_features) + '/GL' + str(year) + '/' + name + '.csv', index = False)










def updateMixtures():
	print('UPDATING MIXTURES')
	for k in list(range(2,8)):
		creatMixtureValues(n_components = k)
		createPCAValues(n_components = k)
		createPitchingPCAValues(n_components = k)
	#gvg
if __name__ == '__main__':
	updateMixtures()
	print('ADDING PLAYER INFO')
	for k in range(2,8):
		add_player_info(num_pitcher_features = k, num_batter_features = k)





'''
k = 2
mod = mixture.BayesianGaussianMixture(n_components = k, max_iter = 100, n_init = 2)
mod.fit(df.values)
a = mod.score(df.values)
#print(a)
#probs.append(a)


a = mod.predict_proba(df.values)

pops = a.sum(axis = 0) / a.shape[0]

a = mod.predict_proba(df.values)[0]
'''

'''
**************************************************
'''


'''
np.random.seed(12345) # set random seed for reproducibility

k = 3
ndata = 500
spread = 5
centers = np.array([1,3, spread])

# simulate data from mixture distribution
v = np.random.randint(0, k, ndata)
data = centers[v] 

plt.hist(data);


model = pm.Model()

with model:
    # cluster sizes
    p = pm.Dirichlet('p', a=np.array([1., 1., 1.]), shape=k)
    # ensure all clusters have some points
    #p_min_potential = pm.Potential('p_min_potential', tt.switch(tt.min(p) < .1, -np.inf, 0))


    # cluster centers
    means = pm.Poisson('means', mu=[1, 1, 1],  shape=k)
    # break symmetry
    order_means_potential = pm.Potential('order_means_potential',
                                         tt.switch(means[1]-means[0] < 0, -np.inf, 0)
                                         + tt.switch(means[2]-means[1] < 0, -np.inf, 0))

    # measurement error
    #sd = pm.Uniform('sd', lower=0, upper=20)

    # latent cluster of each observation
    category = pm.Categorical('category',
                              p=p,
                              shape=ndata)

    # likelihood for each observed value
    points = pm.Poisson('obs',mu=means[category],observed=data)
    print(points)

with model:
    step1 = pm.Metropolis(vars=[p, means])
    step2 = pm.ElemwiseCategorical(vars=[category], values=[0, 1, 2])
    tr = pm.sample(10000, step=[step1, step2])

pm.plots.traceplot(tr, ['p', 'means'])


def cluster_posterior(i=0):
    print('true cluster:', v[i])
    print('  data value:', np.round(data[i],2))
    plt.hist(tr['category'][5000::5,i], bins=[-.5,.5,1.5,2.5,], rwidth=.9)
    plt.axis(xmin=-.5, xmax=2.5)
    plt.xticks([0,1,2])
cluster_posterior()

'''
