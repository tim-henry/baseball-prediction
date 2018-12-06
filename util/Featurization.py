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


def diff_space(df):
    '''
    Put a dataframe into diff space
    '''
    # print(df.columns)
    # player_cols = [col for col in df.columns if col[-1].isdigit() and col[-2].isdigit() and col[-3].isdigit() and len(col) == 8]
    # team_cols = [col for col in player_cols if not col.startswith('opp')]
    # opp_cols = [col for col in player_cols if col.startswith('opp')]

    not_to_diff = ['Season', 'isWin', 'isHome']
    Y = df[not_to_diff]
    # P = df[player_cols]
    # try:
    #     X = df.drop(['isWin', 'isHome', 'Unnamed: 0', 'Name', 'opp_Name'], axis=1)
    # except ValueError:
    # X = df.drop(not_to_diff+player_cols, axis=1)
    X = df.drop(not_to_diff, axis=1)

    home_filter = [col for col in X.columns if not col.startswith('opp')]
    opp_filter = [col for col in X.columns if col.startswith('opp')]
    # filter_col = [col for col in df if col.startswith('Home_Starter')]
    # filter_col = [col for col in df[filter_col] if col.endswith('ID')]
    # df = gameDF[filter_col]

    # XTeam = X.iloc[:, 0:nVars]
    # XOpp = X.iloc[:, nVars:]

    XTeam = X[home_filter]
    XOpp = X[opp_filter]

    # print(list(XTeam))

    XDiff = pd.DataFrame(XTeam.values - XOpp.values)
    XDiff.columns = list(XTeam)

    # nDF = pd.concat([Y.reset_index(drop=True), XDiff, P.reset_index(drop=True)], axis=1).drop('cum_GameNum', axis=1)
    nDF = pd.concat([Y.reset_index(drop=True), XDiff], axis=1).drop('cum_GameNum', axis=1)

    return nDF

def featurize_data(dropbox_dir, featured_dir, start_date = None, end_date = None, type_cum = '', std=True):
    """
    Function to standardize data.
    Std = True for standard Gaussian, False for min/max standardization
    Saves concatenated and standardized dataframe in CUM_CONCAT/CUM_CONCAT.csv
    """
    total = pd.DataFrame()
    for year in os.listdir(dropbox_dir+featured_dir):
        bef= True
        aft = True
        curr_year = int(year[-4:])
        if start_date is not None:
            bef = start_date <= curr_year
        if end_date is not None:
            aft = curr_year <= end_date
        if bef and aft:
            print("Working on files in: {}".format(str(curr_year)))
            for team in os.listdir(dropbox_dir + featured_dir + year):
                data_filename = dropbox_dir + featured_dir + year + '/' + team
                with open(data_filename) as f:
                    team_df = pd.read_csv(f).dropna()
                    to_std = team_df[np.array(cumColNames)]
                    vals_to_std = to_std.values
                    if std:
                        scaler = preprocessing.StandardScaler()
                    else:
                        scaler = preprocessing.MinMaxScaler()
                    vals_scaled = scaler.fit_transform(vals_to_std)
                    team_df['Season'] = curr_year
                    team_df[np.array(cumColNames)] = vals_scaled
                    total = total.append(team_df)

    cols_to_keep = ['Season', 'isWin', 'isHome'] + cumColNames + [col for col in total.columns if col[-1].isdigit() and col[-2].isdigit() and col[-3].isdigit() and (len(col) == 8 or len(col) == 12)]
    # print(cols_to_keep)
    total = total[cols_to_keep]
    # cols_to_drop = ['cum_isWin', 'cum_isHome', 'cum_GameNum',
    #                 'opp_cum_isWin', 'opp_cum_isHome', 'opp_cum_GameNum']
    # total = total.drop(cols_to_drop)
    total = diff_space(total)
    df_title = 'CUM_CONCAT_{}_{}_{}'.format(type_cum, str(start_date) if start_date is not None else '', str(end_date) if end_date is not None else '')
    total.to_csv(dropbox_dir + "CUM_CONCAT/{}.csv".format(df_title), index=False)


def prepare_data(dropbox_dir, reclean = False, refeature = True):

    # featured_dir = 'CUM/'
    featured_dir = 'data_clean_csv_wins_cumulated_withplayers/'
    start_date = 2010
    end_date = 2017
    type_cum = 'SeasAvgPlayers'
    # if reclean:
    #     drop_data(dropbox_dir, featured_dir)
    if refeature:
        featurize_data(dropbox_dir, featured_dir, start_date = start_date, end_date = end_date, type_cum=type_cum,  std=True)







if __name__ == "__main__":
    dropbox_dir = os.path.expanduser("~/Documents/Dropbox (MIT)/6.867 - NEW/")
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
