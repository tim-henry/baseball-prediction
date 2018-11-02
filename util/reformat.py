import pandas as pd
from os import listdir, mkdir
from os.path import isfile, join

# ==========================================================================================================

fields = {"visiting_team": 3, "home_team": 6}

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
