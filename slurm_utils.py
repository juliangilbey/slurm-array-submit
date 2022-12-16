#!/usr/bin/env python3

"""Functions to simplify running parameterised sbatch array jobs.

Can be called to produce total count of all possible parameters and
the nth product.
"""

import argparse
import os
import tempfile
import warnings
from collections import OrderedDict
from functools import reduce
from operator import mul

import slurm_toml_decoder as toml


def get_count(params):
    return count_product(list(params.values()))


def read_toml(filename):
    with open(filename, "r", encoding="UTF-8") as tomlfile:
        toml_dict = toml.load(tomlfile, _dict=OrderedDict)
    return toml_dict


def read_paramfile(filename):
    params = read_toml(filename)
    validated_params = validate_param_dict(params)
    return validated_params


def validate_param_dict(dict_lists):
    """Validate data read from a TOML parameters file.

    Every item should have the form str: list
    The function returns the valid entries.
    """
    outdict = OrderedDict()
    for key, val in dict_lists.items():
        if not isinstance(key, str):
            warnings.warn(
                f"Invalid key in TOML file (not a string): {key}: {val}",
                UserWarning,
            )
            continue
        if not isinstance(val, list):
            warnings.warn(
                f"Invalid value in TOML file (not a list): {key}: {val}",
                UserWarning,
            )
            continue
        outdict[key] = val

    return outdict


def count_product(lists):
    """The product of the lengths of the iterables in the list of lists."""
    lengths = list(map(len, lists))

    if len(lengths) == 0:
        return 1

    return reduce(mul, lengths)


def nth_product(index, lists):
    """Equivalent to ``list(product(params.values()))[index]``.

    The elements of the product of `params` can be ordered lexicographically.
    This function computes the element at sort position *index* without
    computing the previous products.

        >>> nth_product(8, [range(2), range(2), range(2), range(2)])
        (1, 0, 0, 0)

    ``IndexError`` will be raised if the given *index* is invalid.

    This code is taken from `more_itertools`; this saves us from an external
    dependency.  License: MIT
    """
    pools = list(map(tuple, reversed(lists)))
    lengths = list(map(len, pools))

    if len(lengths) == 0:
        return tuple()

    total = reduce(mul, lengths)

    if index < 0:
        index += total

    if not 0 <= index < total:
        raise IndexError

    result = []
    for pool, length in zip(pools, lengths):
        result.append(pool[index % length])
        index //= length

    return tuple(reversed(result))


def get_array_argument(template, paramfile):
    params = read_paramfile(paramfile)
    count = get_count(params)
    # get_count() returns the size of the product, but slurm is 0-based,
    # so if the product size is 60, we need to specify 0-59.
    count -= 1
    array_argument = template.format(count=count)
    return array_argument


def overrides_to_dict(overrides):
    out = OrderedDict()
    for override in overrides:
        keyval = override.split("=", maxsplit=1)
        if len(keyval) < 2:
            warnings.warn(
                f"--sbatch option not of form k=v, ignoring: {override}",
                UserWarning,
            )
        else:
            out[keyval[0]] = keyval[1]

    return out


def get_sbatch_config(configfile, overrides):
    config_base = read_toml(configfile)
    config = {str(k): str(v) for k, v in config_base.items()}
    if overrides is not None:
        overrides_dict = overrides_to_dict(overrides)
        config.update(overrides_dict)
    return config


def substitute_nth(nth, params, command):
    nth_values = nth_product(nth, params.values())
    nth_dict = dict(zip(params.keys(), nth_values))
    return command.format_map(nth_dict)


def make_sbatch_headers(config):
    outstr = ""
    for key, val in config.items():
        if len(key) == 1:
            outstr += f"#SBATCH -{key} {val}\n"
        else:
            outstr += f"#SBATCH --{key}={val}\n"

    return outstr


def make_batchfile_contents(setup, command, paramfile, config):
    # pylint: disable=unused-argument, possibly-unused-variable
    contents_template = """#!/bin/bash
{sbatch_config}

cd {pwd}

command=$(python3 {slurm_utils_dir}/slurm_utils.py --nth=$SLURM_ARRAY_TASK_ID \
          --paramfile={paramfile} --command="{command}")

{setup}

$command
"""

    sbatch_config = make_sbatch_headers(config)
    pwd = os.getcwd()
    slurm_utils_dir = os.path.dirname(__file__)
    contents = contents_template.format_map(locals())

    return contents


def write_sbatch_file(batchfile_contents, sbatch_filename):
    if sbatch_filename is None:
        sbatch_filename = write_sbatch_temporary_file(batchfile_contents)
    else:
        write_sbatch_named_file(batchfile_contents, sbatch_filename)

    return sbatch_filename


def write_sbatch_named_file(batchfile_contents, sbatch_filename):
    with open(sbatch_filename, "w", encoding="UTF-8") as outfile:
        print(batchfile_contents, end="", file=outfile)


def write_sbatch_temporary_file(batchfile_contents):
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="UTF-8", delete=False
    ) as outfile:
        sbatch_filename = outfile.name
        print(batchfile_contents, end="", file=outfile)

    return sbatch_filename


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paramfile",
        help="The parameters TOML filename",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--count",
        help="Return the total number of parameter options from the paramfile",
        action="store_true",
    )
    parser.add_argument(
        "--nth",
        help="Substitute the nth parameter option into the command string",
        type=int,
    )
    parser.add_argument(
        "--command",
        help="String with parameters indicated as {dataset}, to be "
        "substituted when called with --nth",
        type=str,
    )

    return parser


def parse_args():
    parser = build_parser()
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    params = read_paramfile(args.paramfile)
    if args.count:
        print(get_count(params))
    elif args.nth is not None:
        if args.command is None:
            raise ValueError("Need to specify a command when using --nth")
        print(substitute_nth(args.nth, params, args.command))


if __name__ == "__main__":
    main()
