#!/bin/bash
#SBATCH --account=project_462000183

#SBATCH --time=16:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=6
#SBATCH --cpus-per-task=63
#SBATCH --mem=0


#SBATCH --partition=standard-g

#SBATCH --mail-user=maxim.smirnov@aalto.fi
#SBATCH -o ./slurm/%j.out


##echo interactive: srun --account=project_462000183 --partition=standard-g --time=4:00:00 --gpus-per-node 1 --cpus-per-task=4 --mem=32G --pty bash
##echo mkdir tykky_env my-python-env
##echo conda-containerize new --mamba --prefix tykky_env tykky_env_lumi.yml

export WANDB_API_KEY="3a78a74ac0c606c8259b81fca78033b28223a484"

export TMPDIR="/tmp" # MEM storage
rm -rf $TMPDIR/smirnom3_data
mkdir /tmp/smirnom3_data
tar -xf movi_a.tar.gz -C $TMPDIR/smirnom3_data
#tar -xf KITTI_VAL.tar.gz -C $TMPDIR/smirnom3_data
echo "Copied files to temp"


#module purge
module load LUMI/22.08  partition/G
module load PyTorch/1.12.1-cpeGNU-22.08 # move down?
#module load lumi-container-wrapper
#echo $PYTHONPATH
#export PATH="/users/smirnom3/probable-motion/tykky_env/bin:$PATH"
export PYTHONUSERBASE="/users/smirnom3/probable-motion/my-python-env"
export PATH="/users/smirnom3/probable-motion/my-python-env/lib/python3.9/site-packages/:$PATH"


cd src
ln -s $TMPDIR/smirnom3_data data
export TRY_DETERMISM_LVL=0
python main.py --config config_raft_lumi.yaml UNSUPVIDSEG.DATASET MOVi_A

