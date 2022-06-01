"""
Usage:
    create_results.py [--input_file=<input_file>] [--output_file=<output_file>]

"""
import pandas as pd
from docopt import docopt
from collections import defaultdict
import os


def sanitise_path(path):
    path = path.split("/")
    path = path[4:]
    path = "/".join(path)
    return path


def convert(df):
    fin_dict = defaultdict(list)
    for _, row in df.iterrows():
        tmp_dict = {
            "type": row["event_label"].upper(),
            "time-range": [row["onset"], row["offset"]],
        }
        fin_dict[row["filename"]].append(tmp_dict)

    df_ans = pd.DataFrame()
    filenames = list(fin_dict.keys())
    tag = list(fin_dict.values())
    df_ans["path"] = filenames
    df_ans["path"] = df_ans["path"].apply(sanitise_path)
    df_ans["tag"] = tag
    return df_ans


if __name__ == "__main__":
    args = docopt(__doc__)
    df = pd.read_csv(args["--input_file"])
    df = convert(df)
    df.to_csv(args['--output_file'], index=False)
