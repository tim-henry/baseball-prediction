# dm.py

import pandas as pd
import numpy as np
from os import listdir, mkdir
from os.path import isfile, join

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

meta = ['Name', 'GameNum']
#'SacrificeFlies', deleted
data = ['Runs', 'hits', 'doubles', 'triples', 'Homeruns', 'RBI', 'SacrificeHits',  'hit-by-pitch', 'walks', 'LeftOnBase', 'doublePlays', 'at-bats', 'errors']
# data = ['runs', 'hits', 'singles', 'doubles', 'triples', 'home_runs', 'rbi', 'sac_hits', 'sac_flys', 'hbp', 'walks',
#         'left_on_base', 'double_plays', 'at_bats', 'errors']
# team = data.copy()
home_out_cols = ['Home_' + x for x in meta] \
           + ['Visitor_' + x for x in meta] \
           + ['Home_' + x for x in data] \
           + ['Visitor_' + x for x in data]

visitor_out_cols = ['Visitor_' + x for x in meta] \
           + ['Home_' + x for x in meta] \
           + ['Visitor_' + x for x in data] \
           + ['Home_' + x for x in data]

out_col_names = meta \
           + ['opp_' + x for x in meta] \
           + data \
           + ['opp_' + x for x in data]


def nameCols(df):
    df.columns = colNames
    return (df)


# def filter(df, by='team', ID='NYA'):
#
#     if by == 'team':
#         toKeepList = []
#         opponentList = []
#         for row in range(0, df.shape[0]):
#             if df.iloc[row, 3] == ID or df.iloc[row, 6] == ID:
#                 toKeepList.append(True)
#                 if df.iloc[row, 3] == ID:
#                     opponentList.append(df.iloc[row, 6])
#                 if df.iloc[row, 6] == ID:
#                     opponentList.append(df.iloc[row, 3])
#
#         df = df.iloc[toKeepList,]
#
#         df.insert(loc=1, column='Team', value=ID)
#         df.insert(loc=2, column='Opponent', value=opponentList)

def clean_season(season, out_dir, team_name):
    clean_data = pd.DataFrame(data=np.full((season.shape[0], len(home_out_cols)), np.nan, dtype=np.double))
    clean_data.columns = out_col_names

    for index, row in season.iterrows():
        # print(season.loc[1, :])
        # print(season.loc[1, "Home_Name"])
        clean_data.loc[index, :] = season.loc[index, home_out_cols] \
            if season.loc[index, "Home_Name"] == team_name else season.loc[index, home_out_cols]

    clean_data.to_csv("out.txt", index=False)


def clean_season_file(filename, out_dir, team_name):
    clean_season(nameCols(pd.read_csv(filename)), out_dir, team_name)

def parse_all(in_dir, out_dir):
    for year in listdir(in_dir):
        path = join(in_dir, year)
        if not isfile(path):
            for f in listdir(path):
                full_name = join(in_dir, year, f)
                # if isfile(full_name): # TODO ?
                team_name, ext = f.split(".")
                # ignore hidden files, etc.
                if ext.lower() != "txt":
                    continue
                # year = f.split(".")[0]  # remove extension
                # new_dir = join(out_dir, name)
                # mkdir(new_dir)
                clean_season_file(full_name, out_dir, team_name)


# ==========================================================================================================


if __name__ == "__main__":
    dropbox_dir = "/Users/timhenry/Dropbox (MIT)/6.867/"
    parse_all(dropbox_dir + "data_raw_team", "")
