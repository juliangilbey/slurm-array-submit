#!/usr/bin/env python3

"""
Set up an sbatch array script and run it with the specified settings.
"""

import argparse
import subprocess

from slurm_utils import (
    get_array_argument,
    get_sbatch_config,
    make_batchfile_contents,
    write_sbatch_file,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paramfile",
        help="Parameters TOML filename",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--setup",
        help="Setup code to run in the sbatch script before the main command",
        type=str,
    )
    parser.add_argument(
        "--command",
        help="Main (single) command to run; should use parameters such as "
        "{dataset} to be substituted in each run",
        type=str,
    )
    parser.add_argument(
        "--array",
        help="Override the default sbatch --array option 0-{count}",
        type=str,
        default="0-{count}",
    )
    parser.add_argument(
        "--configfile",
        help="sbatch configuration TOML filename, default sbatch-config.toml",
        type=str,
        default="sbatch-config.toml",
    )
    parser.add_argument(
        "--sbatch",
        help="Set an #SBATCH option; for example: --sbatch p=mynode "
        "(can be used multiple times; overrides config file defaults)",
        type=str,
        action="append",
    )
    parser.add_argument(
        "--sbatch-filename",
        help="Specify the filename for the generated sbatch file; by default, "
        "a temporary file is used",
        type=str,
    )

    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    array_argument = get_array_argument(args.array, args.paramfile)
    sbatch_config = get_sbatch_config(args.configfile, args.sbatch)
    batchfile_contents = make_batchfile_contents(
        args.setup, args.command, args.paramfile, sbatch_config
    )
    sbatch_filename = write_sbatch_file(
        batchfile_contents, args.sbatch_filename
    )
    if args.sbatch_filename is None:
        print(f"Wrote temporary sbatch file to {sbatch_filename}")

    subprocess.run(
        ["sbatch", f"--array={array_argument}", sbatch_filename], check=True
    )


if __name__ == "__main__":
    main()
