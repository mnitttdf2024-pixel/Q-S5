#!/bin/bash
#SBATCH -p g24
#SBATCH --account=gts-smukhopadhyay6-ece
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH -t 24:00:00
#SBATCH --qos=high

source ./venv/bin/activate
cd /storage/home/hcoda1/2/apadhy9/Documents/Q-S5/S5fork

python run_qtrain.py \
    --run_name=retrieval-w4a8-ssm8 --checkpoint_dir=/storage/home/hcoda1/2/apadhy9/Documents/Q-S5/final \
    --mlflow_tracking_uri="http://isl-cpu1.rr.intel.com:2517/" --mlflow_experiment_id=676608297636244909 \
    --job_id=$SLURM_JOB_ID \
    --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --ssm_act_bits=8 \
    --non_ssm_bits=4 --non_ssm_act_bits=8 \
    --C_init=trunc_standard_normal --batchnorm=True --bidirectional=True \
    --blocks=16 --bsz=32 --d_model=128 --dataset=aan-classification \
    --dt_global=True --epochs=20 --jax_seed=5464368 --lr_factor=2 --n_layers=6 \
    --opt_config=standard --p_dropout=0.0 --ssm_lr_base=0.00075 --ssm_size_base=256 \
    --warmup_end=1 --weight_decay=0.05 \
    --qgelu_approx=True --hard_sigmoid=True