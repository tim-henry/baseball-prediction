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



def drop_data(dropbox_dir, featured_dir):
    """
    Here we erase statistics that are not game-level. Includes umpire information, pitcher/batter information,
    manager information. Want to focus purely on game-level stats for our model.

    TODO: Instead of just saving these vanilla, want to create a single dataframe for each season. Make sure not
    to double-count games by assigning a new column, gameID.
        This column should be concatenation of Date, Visitor_name, home_name, game_num.
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
    to_one_hot = [1, 2, 3, 4, 5, 7, 8, 13, 15, 16, 17]
    # to_one_hot = [0, 1, 2, 3, 4, 6, 7, 12, 14, 15, 16] #list of categorical columns. From data description list

    for year in os.listdir(dropbox_dir+featured_dir):
        print("Working on files in: {}".format(year))
        for team in os.listdir(dropbox_dir + featured_dir + year):
            data_filename = dropbox_dir + featured_dir + year + '/' + team
            # print(data_filename)
            with open(data_filename) as f:
                team_df = pd.read_csv(f, names=colNames)

                dropped_df = team_df.drop(np.array(colNames[to_one_hot]), axis=1)
                vals_to_std = dropped_df.values
                if std:
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()
                vals_scaled = scaler.fit_transform(vals_to_std)
                std_df = pd.DataFrame(vals_scaled, columns = dropped_df.columns)

                team_df.to_csv(data_filename)



def prepare_data(dropbox_dir, reclean = True, refeature = True):

    featured_dir = 'feature_data_team/'
    if reclean:
        drop_data(dropbox_dir, featured_dir)
    if refeature:
        featurize_data(dropbox_dir, featured_dir)











if __name__ == "__main__":
    dropbox_dir = os.path.expanduser("~/Documents/Dropbox/6.867/")
    # dropbox_dir = os.path.expanduser("~")
    # year = '/GL2017'
    # team = '/ANA.txt'
    with open(dropbox_dir + "data_raw_team/GL1871/BS1.txt") as f:
        team = pd.read_csv(f)
        print(team.shape)
    # print(os.listdir(dropbox_dir))
    prepare_data(dropbox_dir, refeature = False)
