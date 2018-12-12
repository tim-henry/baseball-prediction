import pandas as pd
from os import listdir, mkdir
from os.path import isfile, join

# ==========================================================================================================

fields = {"visiting_team": 3, "home_team": 6}
colNames = ['Date','GameNumber', 'Day', 'Visitor', 'VisitorLeague','VisitorGameNum','Home','HomeLeague','HomeGameNum','VisitorScore','HomeScore','GameLength','Day/Night','CompletionInf','ForfeitInf','ProtestInf','ParkID','Attendance','TimeOfGame','Vistor_LineScores','Home_LineScores','Visitor_at-bats','Visitor_hits','Visitor_doubles','Visitor_triples','Visitor_Homeruns','Visitor_RBI','Visitor_SacrificeHits','Visitor_SacrificeFlies','Visitor_hit-by-pitch','Visitor_walks','Visitor_intentionalWalks','Visitor_strikeouts','Visitor_StolenBases','Visitor_CaughtStealing','Visitor_GroundedIntoDoublePlays','Visitor_AwardedFirstOnCatcherInterference','Visitor_LeftOnBase','Visitor_PitchersUsed','Visitor_IndividualEarnedRuns','Visitor_TeamEarnedRuns','Visitor_WildPitches','Visitor_Balks','Visitor_putouts','Visitor_assists','Visitor_errors','Visitor_PassedBalls','Visitor_doublePlays','Visitor_TriplePlays','Home_at-bats','Home_hits','Home_doubles','Home_triples','Home_Homeruns','Home_RBI','Home_SacrificeHits','Visitor_SacrificeFlies','Home_hit-by-pitch','Home_walks','Home_intentionalWalks','Home_strikeouts','Home_StolenBases','Home_CaughtStealing','Home_GroundedIntoDoublePlays','Home_AwardedFirstOnCatcherInterference','Home_LeftOnBase','Home_PitchersUsed','Home_IndividualEarnedRuns','Home_TeamEarnedRuns','Home_WildPitches','Home_Balks','Home_putouts','Home_assists','Home_errors','Home_PassedBalls','Home_doublePlays','Home_TriplePlays','HomePlateUmpID','HomePlateUmpName','1BUmpID','1BUmpName','2BUmpID','2BUmpName','3BUmpID','3BUmpName','LFUmpID','LFUmpName','RFUmpID','RFUmpName','Visitor_ManagerID','Visitor_ManagerName','Home_ManagerID','Home_ManagerName','WinningPitcherID','WinningPitcherName','LosingPitcherID','LosingPitcherName','SavingPitcherID','SavingPitcherName','GamewinningRBIBatterID','GamewinningRBIBatterName','Visitor_StartingPitcherID','Visitor_StartingPitcherName','Home_StartingPitcherID','Home_StartingPitcherName','Visitor_Starter1ID','Visitor_Starter1Name','Visitor_Starter1DefensivePosition','Visitor_Starter2ID','Visitor_Starter2Name','Visitor_Starter2DefensivePosition','Visitor_Starter3ID','Visitor_Starter3Name','Visitor_Starter3DefensivePosition','Visitor_Starter4ID','Visitor_Starter4Name','Visitor_Starter4DefensivePosition','Visitor_Starter5ID','Visitor_Starter5Name','Visitor_Starter5DefensivePosition','Visitor_Starter6ID','Visitor_Starter6Name','Visitor_Starter6DefensivePosition','Visitor_Starter7ID','Visitor_Starter7Name','Visitor_Starter7DefensivePosition','Visitor_Starter8ID','Visitor_Starter8Name','Visitor_Starter8DefensivePosition','Visitor_Starter9ID','Visitor_Starter9Name','Visitor_Starter9DefensivePosition','Home_Starter1ID','Home_Starter1Name','Home_Starter1DefensivePosition','Home_Starter2ID','Home_Starter2Name','Home_Starter2DefensivePosition','Home_Starter3ID','Home_Starter3Name','Home_Starter3DefensivePosition','Home_Starter4ID','Home_Starter4Name','Home_Starter4DefensivePosition','Home_Starter5ID','Home_Starter5Name','Home_Starter5DefensivePosition','Home_Starter6ID','Home_Starter6Name','Home_Starter6DefensivePosition','Home_Starter7ID','Home_Starter7Name','Home_Starter7DefensivePosition','Home_Starter8ID','Home_Starter8Name','Home_Starter8DefensivePosition','Home_Starter9ID','Home_Starter9Name','Home_Starter9DefensivePosition','AdditionalInfo','AcquisitionInformation']


# ==========================================================================================================


def parse_season(season, out_dir):
    team_to_idx = {}
    for index, row in season.iterrows():
        home = row[fields["home_team"]]
        visitor = row[fields["visiting_team"]]
        if home not in team_to_idx:
            team_to_idx[home] = []
        if visitor not in team_to_idx:
            team_to_idx[visitor] = []

        team_to_idx[home].append(index)
        team_to_idx[visitor].append(index)

    for team in team_to_idx:
        season.iloc[team_to_idx[team], :].to_csv(join(out_dir, team + ".txt"), index=False)


def parse_season_file(filename, out_dir):
    parse_season(pd.read_csv(filename), out_dir)


def parse_all(in_dir, out_dir):
    for f in listdir(in_dir):
        full_name = join(in_dir, f)
        if isfile(full_name):
            name, ext = f.split(".")
            # ignore hidden files, etc.
            if ext != "TXT":
                continue
            year = f.split(".")[0]  # remove extension
            new_dir = join(out_dir, year)
            mkdir(new_dir)
            parse_season_file(full_name, new_dir)


# ==========================================================================================================


if __name__ == "__main__":
    dropbox_dir = "/Users/timhenry/Dropbox (MIT)/6.867/"
    parse_all(dropbox_dir + "data_raw", dropbox_dir + "data_raw_team")
