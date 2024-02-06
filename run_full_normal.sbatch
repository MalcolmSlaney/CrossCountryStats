#!/bin/bash
#SBATCH --job-name=test_plots
#SBATCH --time=900:00
#SBATCH -p normal
#SBATCH --mem=32GB
#SBATCH --mail-type=ALL
#SBATCH -c 1
# below you run/call your code, load modules, python, Matlab, R, etc.
# and do any other scripting you want
# lines that begin with #SBATCH are directives (requests) to the scheduler-SLURM

# To run
#	sbatch run_full_model.sbatch
# To follow queue
#	squeue -u mslaney

ml python/3.9.0
python3 course_stats.py --chains=36 -seed=42 --draws=2000 \
	--cache_dir=Cache.normal --result_dir=Results.normal \
	--monthly_spec=normal,10,10 --yearly_spec=normal,10,10 \
	--course_spec=normal,1,1    --runner_spec=normal,1,0.25
