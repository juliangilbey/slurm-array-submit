import os

import pytest

import slurm_utils


def test_product():
    result = slurm_utils.count_product([[1, 2, 5], ["a", "c"], ["xxx"]])
    assert result == 3 * 2 * 1


def test_nth_product():
    result = slurm_utils.nth_product(2, [[1, 2, 5], ["a", "c"], ["xxx"]])
    assert result == (2, "a", "xxx")


def test_read_toml():
    data = slurm_utils.read_toml("testfiles/params.toml")
    expected = {
        "dataset": ["SYNTHETIC"],
        "imputation": ["MICE", "GAIN", "Mean"],
        "train_percentage": [0.25, 0.5],
        "test_percentage": [0.5],
        "holdout_set": [0, 1, 2],
        "val_set": [0, 1, 2, 3, 4],
        "repeat": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    }
    assert data == expected


def test_read_toml_bad():
    data = slurm_utils.read_toml("testfiles/paramsbad.toml")
    expected = {
        "dataset": ["SYNTHETIC"],
        "imputation": ["MICE", "GAIN", "Mean"],
        "train_percentage": [0.25, 0.5],
        "test_percentage": [0.5],
        "n_holdout_sets": 3,
        "n_val_sets": 5,
        "n_repeats": 10,
    }
    assert data == expected


def test_read_paramfile():
    params = slurm_utils.read_paramfile("testfiles/params.toml")
    expected = {
        "dataset": ["SYNTHETIC"],
        "imputation": ["MICE", "GAIN", "Mean"],
        "train_percentage": [0.25, 0.5],
        "test_percentage": [0.5],
        "holdout_set": [0, 1, 2],
        "val_set": [0, 1, 2, 3, 4],
        "repeat": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    }
    assert params == expected


def test_read_paramfile_bad():
    with pytest.warns(UserWarning):
        params = slurm_utils.read_paramfile("testfiles/paramsbad.toml")
    expected = {
        "dataset": ["SYNTHETIC"],
        "imputation": ["MICE", "GAIN", "Mean"],
        "train_percentage": [0.25, 0.5],
        "test_percentage": [0.5],
    }
    assert params == expected


def test_get_count():
    params = slurm_utils.read_paramfile("testfiles/params.toml")
    count = slurm_utils.get_count(params)
    assert count == 1 * 3 * 2 * 1 * 3 * 5 * 10


@pytest.mark.parametrize(
    "template,expected",
    [
        ("0-{count}", "0-899"),
        ("0-{count}%10", "0-899%10"),
        ("50-{count}%1", "50-899%1"),
    ],
)
def test_get_array_argument(template, expected):
    paramfile = "testfiles/params.toml"
    argument = slurm_utils.get_array_argument(template, paramfile)
    assert argument == expected


def test_get_sbatch_config():
    overrides = ["z=0", "A=OTHER-ACCOUNT"]
    config = slurm_utils.get_sbatch_config(
        "testfiles/sbatch-config.toml", overrides
    )

    result = {
        "p": "mynode",
        "A": "OTHER-ACCOUNT",
        "N": "2",
        "time": "15:00:00",
        "o": "messages_%a.out",
        "z": "0",
    }

    assert config == result


def test_get_sbatch_config_empty_overrides():
    overrides = None
    config = slurm_utils.get_sbatch_config(
        "testfiles/sbatch-config.toml", overrides
    )

    result = {
        "p": "mynode",
        "A": "MY-ACCOUNT",
        "N": "2",
        "time": "15:00:00",
        "o": "messages_%a.out",
    }

    assert config == result


def test_substitute_nth():
    params = slurm_utils.read_paramfile("testfiles/params.toml")
    command = (
        "python3 myscript.py --dataset={dataset} {imputation} {holdout_set} "
        "--val={val_set} --repeat={repeat}"
    )
    # We have:
    # params = {
    #     "dataset": ["SYNTHETIC"],
    #     "imputation": ["MICE", "GAIN", "Mean"],
    #     "train_percentage": [0.25, 0.5],
    #     "test_percentage": [0.5],
    #     "holdout_set": [0, 1, 2],
    #     "val_set": [0, 1, 2, 3, 4],
    #     "repeat": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    # }

    nth = (
        2 * (2 * 3 * 5 * 10)  # MICE and GAIN
        + 5 * 10  # Mean, holdout_set 0
        + 3 * 10  # Mean, holdout_set 1, val_set 0-2
        + 7  # Mean, holdout_set 1, val_set 3, repeat 0-6
    )
    expected = (
        "python3 myscript.py --dataset=SYNTHETIC Mean 1 --val=3 --repeat=7"
    )
    substituted = slurm_utils.substitute_nth(nth, params, command)

    assert substituted == expected


def test_substitute_nth_bad():
    """Substitution should fail with an unrecognised parameter"""
    params = slurm_utils.read_paramfile("testfiles/params.toml")
    command = (
        "python3 myscript.py --dataset={dataset} {imputation} {holdout_set} "
        "--val={val_set} --repeat={repeat} --unknown={unknown}"
    )

    nth = 370
    with pytest.raises(KeyError):
        slurm_utils.substitute_nth(nth, params, command)


def test_make_batchfile_contents():
    setup = """
module load miniconda3
source activate local_python_venv
"""
    command = (
        "python3 longscript.py {dataset} {imputation} "
        "{train_percentage} {test_percentage} {holdout_set} {val_set} "
        "{repeat} {nfeat} {nmar} {nshuff}"
    )
    overrides = None
    config = slurm_utils.get_sbatch_config(
        "testfiles/sbatch-config.toml", overrides
    )
    paramfile = "testfiles/params.toml"

    batchfile = slurm_utils.make_batchfile_contents(
        setup, command, paramfile, config
    )

    pwd = os.getcwd()

    result = """#!/bin/bash
#SBATCH -p mynode
#SBATCH -A MY-ACCOUNT
#SBATCH -N 2
#SBATCH --time=15:00:00
#SBATCH -o messages_%a.out


cd PWD

command=$(python3 PWD/slurm_utils.py --nth=$SLURM_ARRAY_TASK_ID \
          --paramfile=PARAMFILE --command="COMMAND")


module load miniconda3
source activate local_python_venv


$command
"""
    result = result.replace("PWD", pwd)
    result = result.replace("PARAMFILE", paramfile)
    result = result.replace("COMMAND", command)

    assert batchfile == result


def test_write_sbatch_file_named(tmp_path):
    content = """#!/bin/bash
echo This is some test content
"""

    outpath = str(tmp_path / "sbatch.sh")
    returned_filename = slurm_utils.write_sbatch_file(content, outpath)

    with open(outpath, "r", encoding="UTF-8") as written_file:
        written = written_file.read()

    assert written == content
    assert returned_filename == outpath


def test_write_sbatch_file_temporary(tmp_path):
    content = """#!/bin/bash
echo This is some test content
"""

    returned_filename = slurm_utils.write_sbatch_file(content, None)

    with open(returned_filename, "r", encoding="UTF-8") as written_file:
        written = written_file.read()

    assert written == content

    os.remove(returned_filename)
