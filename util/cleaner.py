import pandas as pd
import numpy as np
from os import listdir, mkdir
from os.path import isfile, join


# ==========================================================================================================


colNames = ['Date', 'GameNum', 'Day', 'Visitor_Name', 'Visitor_League', 'Visitor_GameNum', 'Home_Name', 'Home_League',
            'Home_GameNum', 'Visitor_Runs', 'Home_Runs', 'GameLength', 'Day/Night', 'CompletionInf', 'ForfeitInf',
            'ProtestInf', 'ParkID', 'Attendance', 'TimeOfGame', 'Vistor_LineScores', 'Home_LineScores',
            'Visitor_at-bats', 'Visitor_hits', 'Visitor_doubles', 'Visitor_triples', 'Visitor_Homeruns', 'Visitor_RBI',
            'Visitor_SacrificeHits', 'Visitor_SacrificeFlies', 'Visitor_hit-by-pitch', 'Visitor_walks',
            'Visitor_intentionalWalks', 'Visitor_strikeouts', 'Visitor_StolenBases', 'Visitor_CaughtStealing',
            'Visitor_GroundedIntoDoublePlays', 'Visitor_AwardedFirstOnCatcherInterference', 'Visitor_LeftOnBase',
            'Visitor_PitchersUsed', 'Visitor_IndividualEarnedRuns', 'Visitor_TeamEarnedRuns', 'Visitor_WildPitches',
            'Visitor_Balks', 'Visitor_putouts', 'Visitor_assists', 'Visitor_errors', 'Visitor_PassedBalls',
            'Visitor_doublePlays', 'Visitor_TriplePlays', 'Home_at-bats', 'Home_hits', 'Home_doubles', 'Home_triples',
            'Home_Homeruns', 'Home_RBI', 'Home_SacrificeHits', 'Visitor_SacrificeFlies', 'Home_hit-by-pitch',
            'Home_walks', 'Home_intentionalWalks', 'Home_strikeouts', 'Home_StolenBases', 'Home_CaughtStealing',
            'Home_GroundedIntoDoublePlays', 'Home_AwardedFirstOnCatcherInterference', 'Home_LeftOnBase',
            'Home_PitchersUsed', 'Home_IndividualEarnedRuns', 'Home_TeamEarnedRuns', 'Home_WildPitches', 'Home_Balks',
            'Home_putouts', 'Home_assists', 'Home_errors', 'Home_PassedBalls', 'Home_doublePlays', 'Home_TriplePlays',
            'HomePlateUmpID', 'HomePlateUmpName', '1BUmpID', '1BUmpName', '2BUmpID', '2BUmpName', '3BUmpID',
            '3BUmpName', 'LFUmpID', 'LFUmpName', 'RFUmpID', 'RFUmpName', 'Visitor_ManagerID', 'Visitor_ManagerName',
            'Home_ManagerID', 'Home_ManagerName', 'WinningPitcherID', 'WinningPitcherName', 'LosingPitcherID',
            'LosingPitcherName', 'SavingPitcherID', 'SavingPitcherName', 'GamewinningRBIBatterID',
            'GamewinningRBIBatterName', 'Visitor_StartingPitcherID', 'Visitor_StartingPitcherName',
            'Home_StartingPitcherID', 'Home_StartingPitcherName', 'Visitor_Starter1ID', 'Visitor_Starter1Name',
            'Visitor_Starter1DefensivePosition', 'Visitor_Starter2ID', 'Visitor_Starter2Name',
            'Visitor_Starter2DefensivePosition', 'Visitor_Starter3ID', 'Visitor_Starter3Name',
            'Visitor_Starter3DefensivePosition', 'Visitor_Starter4ID', 'Visitor_Starter4Name',
            'Visitor_Starter4DefensivePosition', 'Visitor_Starter5ID', 'Visitor_Starter5Name',
            'Visitor_Starter5DefensivePosition', 'Visitor_Starter6ID', 'Visitor_Starter6Name',
            'Visitor_Starter6DefensivePosition', 'Visitor_Starter7ID', 'Visitor_Starter7Name',
            'Visitor_Starter7DefensivePosition', 'Visitor_Starter8ID', 'Visitor_Starter8Name',
            'Visitor_Starter8DefensivePosition', 'Visitor_Starter9ID', 'Visitor_Starter9Name',
            'Visitor_Starter9DefensivePosition', 'Home_Starter1ID', 'Home_Starter1Name',
            'Home_Starter1DefensivePosition', 'Home_Starter2ID', 'Home_Starter2Name', 'Home_Starter2DefensivePosition',
            'Home_Starter3ID', 'Home_Starter3Name', 'Home_Starter3DefensivePosition', 'Home_Starter4ID',
            'Home_Starter4Name', 'Home_Starter4DefensivePosition', 'Home_Starter5ID', 'Home_Starter5Name',
            'Home_Starter5DefensivePosition', 'Home_Starter6ID', 'Home_Starter6Name', 'Home_Starter6DefensivePosition',
            'Home_Starter7ID', 'Home_Starter7Name', 'Home_Starter7DefensivePosition', 'Home_Starter8ID',
            'Home_Starter8Name', 'Home_Starter8DefensivePosition', 'Home_Starter9ID', 'Home_Starter9Name',
            'Home_Starter9DefensivePosition', 'AdditionalInfo', 'AcquisitionInformation']

meta = ['Name', 'GameNum']

# Note 'SacrificeFlies', deleted, inconsistent # of cols
data =['at-bats', 'hits', 'doubles', 'triples', 'Homeruns', 'RBI',
            'SacrificeHits', 'hit-by-pitch', 'walks',
            'intentionalWalks', 'strikeouts', 'StolenBases', 'CaughtStealing',
            'GroundedIntoDoublePlays', 'AwardedFirstOnCatcherInterference', 'LeftOnBase',
            'PitchersUsed', 'IndividualEarnedRuns', 'TeamEarnedRuns', 'WildPitches',
            'Balks', 'putouts', 'assists', 'errors', 'PassedBalls',
            'doublePlays', 'TriplePlays']

home_out_cols = ['Home_' + x for x in meta] \
           + ['Visitor_' + x for x in meta] \
           + ['isHome'] \
           + ['Home_' + x for x in data] \
           + ['Visitor_' + x for x in data]

visitor_out_cols = ['Visitor_' + x for x in meta] \
           + ['Home_' + x for x in meta] \
           + ['isHome'] \
           + ['Visitor_' + x for x in data] \
           + ['Home_' + x for x in data]

out_col_names = meta \
           + ['opp_' + x for x in meta] \
           + ['isHome'] \
           + data \
           + ['opp_' + x for x in data]


# ==========================================================================================================


def nameCols(df):
    df.columns = colNames
    return df


def clean_season(season, out_dir, team_name):
    n = season.shape[0]

    # append isHome binary indicator
    is_home = pd.DataFrame(data=np.zeros((n, 1)), columns=["isHome"])
    for index, row in season.iterrows():
        is_home.iloc[index] = 1 if season.loc[index, "Home_Name"] == team_name else 0
    season = pd.concat([season, is_home], axis=1)

    # remove/reorganize columns
    clean_data = pd.DataFrame(data=np.full((n, len(home_out_cols)), np.nan, dtype=np.double))
    clean_data.columns = out_col_names

    for index, row in season.iterrows():
        clean_data.loc[index] = row[home_out_cols].tolist() \
            if season.loc[index, "Home_Name"] == team_name else row[visitor_out_cols].tolist()

    clean_data.to_csv(join(out_dir, team_name + ".csv"), index=False)


def clean_season_file(filename, out_dir, team_name):
    clean_season(nameCols(pd.read_csv(filename)), out_dir, team_name)


def parse_all(in_dir, out_dir):
    for year in listdir(in_dir):
        if year == ".DS_Store":
            continue
        new_dir = join(out_dir, year)
        mkdir(new_dir)

        path = join(in_dir, year)
        if not isfile(path):
            for f in listdir(path):
                full_name = join(in_dir, year, f)
                team_name, ext = f.split(".")
                # ignore hidden files, etc.
                if ext.lower() != "txt" and ext.lower() != "csv":
                    continue
                clean_season_file(full_name, new_dir, team_name)


# ==========================================================================================================


if __name__ == "__main__":
    dropbox_dir = "/Users/timhenry/Dropbox (MIT)/6.867/"
    parse_all(dropbox_dir + "data_raw_team", dropbox_dir + "data_clean_csv")