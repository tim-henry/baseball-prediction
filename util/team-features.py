import numpy as np
from os import listdir
from os.path import isfile, join
import pandas as pd


# Load Data
def update_teams(all_teams, in_dir, year):
    seqs = []
    labels = []
    path = join(in_dir, year)
    if not isfile(path):
        for f in listdir(path):
            full_name = join(in_dir, year, f)
            team_name, ext = f.split(".")
            # ignore hidden files, etc.
            if ext.lower() != "txt" and ext.lower() != "csv":
                continue
            raw_data = pd.read_csv(full_name)
            names = raw_data['Name']
            for name in names:
                if name not in all_teams:
                    all_teams.add(name)


if __name__ == '__main__':
    dropbox_dir = "/Users/timhenry/Dropbox (MIT)/6.867/"
    in_dir = dropbox_dir + "data_clean_csv_wins_cumulated"

    all_teams = set()
    for year in range(1871, 2018):
        update_teams(all_teams, in_dir, "GL" + str(year))
    n = len(all_teams)
    print(n, all_teams)
    encoding = pd.DataFrame(0, index=range(n + 1), columns=range(n))
    row_idx, col_idx = 1, n - 1
    encoding = encoding.rename(index={0: "unknown"})
    for team in all_teams:
        encoding.at[row_idx, col_idx] = 1
        encoding = encoding.rename(index={row_idx: team})
        encoding = encoding.rename(columns={col_idx: team})
        row_idx += 1
        col_idx -= 1
    print(encoding.columns)
    encoding.to_csv(dropbox_dir + "teams-one-hot.csv")
