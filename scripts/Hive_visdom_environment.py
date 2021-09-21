#! /usr/bin/env python

import argparse

from Hive.evaluation.vis import create_log, create_log_at, load_log, load_all_log, load_log_at

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and Save from Visdom")
    parser.add_argument("-s", "--save", type=str, help="env_name", default="")
    parser.add_argument("-l", "--load", type=str, help="env_name", default="", nargs="?")
    parser.add_argument("-f", "--file", type=str, help="path_to_log_file", default="")
    args = parser.parse_args()

    if args.save != "":
        if args.file != "":
            create_log_at(args.file, args.save)
        else:
            create_log(args.save)

    if args.load != "":
        if args.load == "all":
            load_all_log()
        elif args.load is not None:
            load_log(args.load)
        elif args.file != "":
            load_log_at(args.file)
