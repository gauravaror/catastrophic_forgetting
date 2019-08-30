#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=1-24:00:00
#SBATCH --partition=deeplearn
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH -A punim0478
#SBATCH -q  gpgpudeeplearn

for try in {1,2,3,4,5,6,7,8,9,10,} ; do
  for layer in {1,2,}; do
    for hidden in {100,400,}; do
      # Launch multiple process python code
      for task in {cola,trec,sst,subjectivity}; do
        for evaluate in {cola,trec,sst,subjectivity}; do
          echo "Running Layer $layer Hidden dimension $hidden" ;
          export CUDA_VISIBLE_DEVICES=0; /home/gaurava/miniconda3/envs/catastrophic/bin/python train.py   --task trec  --task sst --task cola  --task subjectivity  --layers $layer --h_dim $hidden --dropout 0.5  --diff_class --cnn --run_name exper_single_few_shot_${task}_${evaluate}_${try} --tryno $try --train $task  --evaluate $evaluate --few_shot --storage_prefix ./runs/single/;
          find ./runs/single/ -name '*.th' | xargs rm
          find ./runs/single -name '*.json' | xargs rm
        done
      done
    done
  done
done
