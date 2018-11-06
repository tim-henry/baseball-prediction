import numpy as np
import pandas as pd
import os
from sklearn import preprocessing

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

regColNames =   ['Unnamed: 0', 'isWin', 'isHome', 'Name', 'GameNum', 'opp_Name', 'opp_GameNum', 'Runs', 'at-bats', 'hits', 'doubles', 'triples',
                'Homeruns', 'RBI', 'SacrificeHits', 'hit-by-pitch', 'walks', 'intentionalWalks', 'strikeouts', 'StolenBases', 'CaughtStealing',
                'GroundedIntoDoublePlays', 'AwardedFirstOnCatcherInterference', 'LeftOnBase', 'PitchersUsed', 'IndividualEarnedRuns',
                'TeamEarnedRuns', 'WildPitches', 'Balks', 'putouts', 'assists', 'errors', 'PassedBalls', 'doublePlays', 'TriplePlays', 'opp_Runs',
                'opp_at-bats', 'opp_hits', 'opp_doubles', 'opp_triples', 'opp_Homeruns', 'opp_RBI', 'opp_SacrificeHits', 'opp_hit-by-pitch',
                'opp_walks', 'opp_intentionalWalks', 'opp_strikeouts', 'opp_StolenBases', 'opp_CaughtStealing', 'opp_GroundedIntoDoublePlays',
                'opp_AwardedFirstOnCatcherInterference', 'opp_LeftOnBase', 'opp_PitchersUsed', 'opp_IndividualEarnedRuns', 'opp_TeamEarnedRuns',
                'opp_WildPitches', 'opp_Balks', 'opp_putouts', 'opp_assists', 'opp_errors', 'opp_PassedBalls', 'opp_doublePlays', 'opp_TriplePlays',
                'cum_isWin', 'cum_isHome', 'cum_GameNum', 'cum_Runs', 'cum_at-bats', 'cum_hits', 'cum_doubles', 'cum_triples',
               'cum_Homeruns', 'cum_RBI', 'cum_SacrificeHits', 'cum_hit-by-pitch', 'cum_walks', 'cum_intentionalWalks', 'cum_strikeouts',
               'cum_StolenBases', 'cum_CaughtStealing', 'cum_GroundedIntoDoublePlays', 'cum_AwardedFirstOnCatcherInterference', 'cum_LeftOnBase',
               'cum_PitchersUsed', 'cum_IndividualEarnedRuns', 'cum_TeamEarnedRuns', 'cum_WildPitches', 'cum_Balks', 'cum_putouts', 'cum_assists',
               'cum_errors', 'cum_PassedBalls', 'cum_doublePlays', 'cum_TriplePlays', 'opp_cum_isWin', 'opp_cum_isHome', 'opp_cum_GameNum',
               'opp_cum_Runs', 'opp_cum_at-bats', 'opp_cum_hits', 'opp_cum_doubles', 'opp_cum_triples', 'opp_cum_Homeruns', 'opp_cum_RBI',
               'opp_cum_SacrificeHits', 'opp_cum_hit-by-pitch', 'opp_cum_walks', 'opp_cum_intentionalWalks', 'opp_cum_strikeouts',
               'opp_cum_StolenBases', 'opp_cum_CaughtStealing', 'opp_cum_GroundedIntoDoublePlays', 'opp_cum_AwardedFirstOnCatcherInterference',
               'opp_cum_LeftOnBase', 'opp_cum_PitchersUsed', 'opp_cum_IndividualEarnedRuns', 'opp_cum_TeamEarnedRuns', 'opp_cum_WildPitches',
               'opp_cum_Balks', 'opp_cum_putouts', 'opp_cum_assists', 'opp_cum_errors', 'opp_cum_PassedBalls', 'opp_cum_doublePlays',
               'opp_cum_TriplePlays']

cumColNames = [cn for cn in regColNames if 'cum' in cn]

def drop_data(dropbox_dir, featured_dir):
    """
    DEPRECATED. DON'T USE
    Here we erase statistics that are not game-level. Includes umpire information, pitcher/batter information,
    manager information. Want to focus purely on game-level stats for our model.
    """

    raw_data_dir = 'data_raw_team/'
    to_drop = [i for i in range(78, 160)]


    for year in os.listdir(dropbox_dir+raw_data_dir):
        print("Working on files in: {}".format(year))
        year_df = pd.DataFrame()
        out_dir = dropbox_dir + featured_dir + year + '.txt'
        for team in os.listdir(dropbox_dir + raw_data_dir + year):
            data_filename = dropbox_dir + raw_data_dir + year + '/' + team
            # print(data_filename)
            with open(data_filename) as f:
                team_df = pd.read_csv(f, names = colNames)
                #Drop the relevant columns from the dataframe.
                team_df = team_df.drop(np.array(colNames)[to_drop], axis=1)
                # team_df['GAMEID'] = team_df['Date'] + '_' + team_df['Visitor_Name'] + '_' + team_df['Home_Name'] + '_' + team_df['GameNum']
                team_df['Date'] = pd.to_datetime(team_df['Date'], format = '%Y%m%d')
                #WRITE TO OUT DATAFRAME.
                # if not os.path.exists(out_dir):
                #     os.makedirs(out_dir)
                # team_df.to_csv(out_dir + team)
                year_df = year_df.append(team_df)
        year_df = year_df.drop_duplicates()
        year_df.to_csv(out_dir)

def featurize_data(dropbox_dir, featured_dir, std=True):
    """
    Function to standardize data and turn categorical variables into one-hot encodings.
    STD: boolean to control whether data is normalized to std. Gaussian or minmax normalization.

    """
    # to_one_hot = [1, 2, 3, 4, 5, 7, 8, 13, 15, 16, 17]
    # to_one_hot = [0, 1, 2, 3, 4, 6, 7, 12, 14, 15, 16] #list of categorical columns. From data description list
    total = pd.DataFrame()
    for year in os.listdir(dropbox_dir+featured_dir):
        print("Working on files in: {}".format(year))
        for team in os.listdir(dropbox_dir + featured_dir + year):
            data_filename = dropbox_dir + featured_dir + year + '/' + team
            # print(data_filename)
            with open(data_filename) as f:
                team_df = pd.read_csv(f).dropna()

                # dropped_df = team_df.drop(np.array(colNames[to_one_hot]), axis=1)
                to_std = team_df[np.array(cumColNames)]
                vals_to_std = to_std.values
                if std:
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()
                vals_scaled = scaler.fit_transform(vals_to_std)
                team_df[np.array(cumColNames)] = vals_scaled
                # std_df = pd.DataFrame(vals_scaled, columns = to_std.columns)
                # print(team_df.head(1))
                total = total.append(team_df)
                # team_df.to_csv(data_filename)
    cols_to_keep = ['isWin', 'isHome'] + cumColNames
    # print(list(total.columns))
    # print(total[cols_to_keep])
    total = total[cols_to_keep]
    cols_to_drop = ['cum_isWin', 'cum_isHome', 'cum_GameNum',
                    'opp_cum_isWin', 'opp_cum_isHome', 'opp_cum_GameNum']
    total = total.drop(cols_to_drop)
    total.to_csv(dropbox_dir + "CUM_CONCAT/CUM_CONCAT.csv")


def prepare_data(dropbox_dir, reclean = False, refeature = True):

    featured_dir = 'CUM/'
    if reclean:
        drop_data(dropbox_dir, featured_dir)
    if refeature:
        featurize_data(dropbox_dir, featured_dir)







if __name__ == "__main__":
    dropbox_dir = os.path.expanduser("~/Documents/Dropbox/6.867/")
    # dropbox_dir = os.path.expanduser("~")
    # year = '/GL2017'
    # team = '/ANA.txt'
    # with open(dropbox_dir + "data_raw_team/GL1871/BS1.txt") as f:
    #     team = pd.read_csv(f)
    #     print(team.shape)
    # print(os.listdir(dropbox_dir))
    prepare_data(dropbox_dir, reclean = False, refeature = True)
    # f = open(dropbox_dir + 'CUM/DUMMY/ANA.csv')
    # team = pd.read_csv(f)
    # print(team.shape)
    # print(team[cumColNames])
    # print(cumColNames)

