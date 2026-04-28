#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import numpy
import pandas
import argparse
import glob


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Aggregate CSV data files with the format Word | Freq. | Statistic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--input-dir", "-i", required=True, help="Input directory path")
    parser.add_argument(
        "--id-columns",
        required=False,
        nargs="+",
        default=None,
        help="Data columns that are invariant (e.g., words). Default: all but last column.",
    )
    parser.add_argument(
        "--var-columns",
        required=False,
        nargs="+",
        default=None,
        help="Data columns that are variable (e.g., statistical indicator or results). Default: last column.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    input_dir = args.input_dir
    all_csv_files = glob.glob(input_dir + "/*.csv")

    aggregated_df = pandas.DataFrame()

    for file in all_csv_files:
        df = pandas.read_csv(file)
        if isinstance(args.id_columns, type(None)):
            args.id_columns = list(df.columns)[:-1]
            print(f"Using {args.id_columns} as ID columns")
        if isinstance(args.var_columns, type(None)):
            args.var_columns = list(df.columns)[-1]
            print(f"Using {args.var_columns} as variable columns")

        videos = file[file.find("_video_") + 7 : file.find(".csv")].split("_")
        if len(videos) > 1:
            video_prefix = "videos_"
        else:
            video_prefix = "video_"

        ### Note of caution: the words are not ordered in the same way in each CSV file!
        if file == all_csv_files[0]:
            aggregated_df[args.id_columns] = df[args.id_columns]
            if isinstance(args.var_columns, list):
                for i in range(len(args.var_columns)):
                    aggregated_df[f"var_{i}"] = args.var_columns[i]
                    aggregated_df[video_prefix + "_".join(videos) + f"_val_{i}"] = df[
                        args.var_columns[i]
                    ]

            else:
                aggregated_df["var"] = args.var_columns
                aggregated_df[video_prefix + "_".join(videos)] = df[args.var_columns]

            ### Reindex the data using the ID columns (to ensure the added data falls in the right place)
            aggregated_df = aggregated_df.set_index(args.id_columns)

        else:
            ### aggregated_df should have been reindexed at this point
            if isinstance(args.var_columns, list):
                for i in range(len(args.var_columns)):
                    aggregated_df[video_prefix + "_".join(videos) + f"_val_{i}"] = (
                        df.set_index(args.id_columns)[args.var_columns[i]]
                    )

            else:
                aggregated_df[video_prefix + "_".join(videos)] = df.set_index(
                    args.id_columns
                )[args.var_columns]

    ### Once all the data has been added, reset the index
    aggregated_df = aggregated_df.reset_index()

    print("\nCompiled DataFrame:")
    print(aggregated_df)

    print(f"""\nSaving to CSV: {args.input_dir + '/aggregated_data.csv.txt'}""")
    aggregated_df.to_csv(args.input_dir + "/aggregated_data.csv.txt", index=False)


if __name__ == "__main__":
    main()
