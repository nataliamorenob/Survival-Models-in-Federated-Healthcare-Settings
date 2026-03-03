#!/bin/bash

N_RUNS=10

cd /scratch/project_2015651/Masters_thesis
source venv/bin/activate

mkdir -p logs
mkdir -p results_randomness_exps

for i in $(seq 1 $N_RUNS); do
    echo "=== RUN $i ==="
    RUN_ID=$i OUTPUT_CSV=results_randomness_exps/run_${i}.csv \
        python src/main.py \
        2>&1 | tee logs/run_${i}.log
done

#chmod +x Exps_runs_randomness/exps_seeds.sh
#./Exps_runs_randomness/exps_seeds.sh
