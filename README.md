# The slurm-array-submit scripts

When performing lots of relatively small experiments on an HPC which
differ only in their parameters, it can be better to run the
experiment with each set of parameters as a separate task, rather than
have one very large task for the entire set of experiments.  This
small collection of scripts grew out of this realisation.  It is
designed for use with the [Slurm](https://slurm.schedmd.com/) workload
manager.

It allows one to easily submit an array of tasks by writing a single
template script and letting these scripts do the rest of the work.
For example, suppose we are testing different classifiers on some
datasets, using all possible combinations of various parameters.  The
following script does this:

```
#!/bin/bash

param_file=params-myexperiment.txt

cat >$param_file <<EOF
dataset = ["NICE-DATA", "NASTY-DATA"]
classifier = ["NGBoost", "XGBoost", "RandomForest"]
depth = [50, 60, 70, 80]
EOF

setup='
module load miniconda3
. my_python_env/bin/activate
'

command='python do_classification.py --dataset={dataset} --classifier={classifier} --depth={depth}'

./slurm_sbatch.py --paramfile=$param_file --setup="$setup" --command="$command"
```

This calcuates the number of parameter combinations, creates a script
for `sbatch` to run, and calls `sbatch` on it.  It reads the slurm
configuration parameters (which are supplied as `#SBATCH` directives
within the script) from a configuration file or from the command line.

The present incarnation of these scripts requires the three Python
scripts and one configuration file to be copied into a suitable
directory to avoid having to install a Python package.  The code is
designed to work with Python 3.6 and it has no dependencies.

## A note about the Slurm array facility

Slurm allows a user to submit an array job with the `sbatch` command.
This will schedule a number of tasks, with each task running an
identical command.  For example:

```
sbatch --array=0-999 mytask.sh
```

will run `mytask.sh` 1000 times.  Within `mytask.sh`, the environment
variable `$SLURM_ARRAY_TASK_ID` is set to the task ID (between 0 and
999 in this case); within `#SBATCH` directives, this is available as
`%a`.

These scripts automatically determine which set of parameter values
should be used for a particular task ID.

## Usage of the `slurm_sbatch.py` script

The `slurm_sbatch.py` script reads the sbatch configuration file
`sbatch-config.toml` (written in TOML format), which is likely to be
common to multiple sbatch calls, along with a parameter file and
command specified on the command line, and generates and runs an
appropriate sbatch script.

The generated sbatch script has the form:

```
#!/bin/bash

#SBATCH [sbatch option]
#SBATCH [sbatch option]
...

[setup commands]

[main command]
```

(It is saved as a temporary file unless the `--sbatch-filename`
option is specified.)

The sbatch options are specified in the file `sbatch-config.toml`,
which should contain lines such as:

```
p = "mynode"
A = "MY-ACCOUNT"
N = 1
time = "12:00:00"
o = "messages_%a.out"
e = "messages_%a.err"
```

Note that the TOML format requires strings to be enclosed in quotes.
Times should be specified as strings to prevent them from being
interpreted.  An alternative option file can be specified using the
`--configfile` option, and individual options can be added or
configuration file options overridden using the `--sbatch` option.

The setup commands are specified using the `--setup` option; this
takes a single string (possible contained embedded newlines) which
will be inserted verbatim into the batch file.

The main command, specified as the `--command` option, can contain
parameters enclosed in braces, for example `'run_experiment.py
--iterations={iterations} --network={network}'`.  (This has to be
quoted so that the shell interprets it as the single argument of
`--command`.)  These parameters are substituted by the appropriate
values based on the parameters file (another TOML file specified by
`--paramfile`).  In this case, the parameter file might read:

```
network = ["VGG-16", "ResNet-18", "ResNet-50"]
iterations = [100, 200, 300, 400, 500]
```

There should be no subsections in the TOML file, and each entry must
be a list of possible values of the parameter.

`slurm_sbatch.py` will determine the number of runs required to use
every possible parameter combination (15 in this example) and will run
the generated script described above that number of times using an
`sbatch` job with `--array=0:14`.  This sbatch array option can be
overridden using the `slurm_sbatch.py` `--array` option, for example
`--array=0-{count}%4`, where `{count}` is replaced by the maximum task
ID (which is one less than the number of parameter combinations, 14 in
this case).
