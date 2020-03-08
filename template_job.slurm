#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=2-12:00:00
#SBATCH --partition=deeplearn
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH -A punim0478 
#SBATCH -q gpgpudeeplearn

first_task=(trec)
others=(trec sst cola subjectivity)
others=${others[@]/$first_task}
run_name='s_8m_transformer_emb_temp_inc_exper_'
storage_prefix="./runs/${run_name}/${first_task}/"
dropout=0.5
patience=10
epochs=100
arr_hidden="100 400"
arr_layer="1 2"
arr_try="1 2 3 4 5"
bs=1024
inv_temp=5
temp_inc=2

run_configuration() {
  first_task=$1
  second_task=$2
  third_task=$3
  fourth_task=$4
  layer=$5
  hidden=$6
  try=$7

  export CUDA_VISIBLE_DEVICES=0; /home/gaurava/miniconda3/envs/catastrophic/bin/python train.py   --task $first_task  --task  $second_task --task $third_task  --task  $fourth_task  --layers $layer --h_dim $hidden --dropout $dropout --run_name $run_name --tryno $try --epochs $epochs --patience $patience --storage_prefix $storage_prefix --bs $bs --transformer --inv_temp 5 --temp_inc 2 --no_save_weight >> ${run_name}_${first_task}.log;
  find $storage_prefix -name '*.pth' | xargs rm

}

for hidden in ${arr_hidden}; do
  for layer in ${arr_layer}; do
    for try in ${arr_try}; do
      # Launch multiple process python code
      echo "Running Layer $layer Hidden dimension $hidden" ;
      run_configuration $first_task  ${others[3]} ${others[1]} ${others[2]} $layer $hidden $try
      run_configuration $first_task  ${others[1]} ${others[3]} ${others[2]} $layer $hidden $try
      run_configuration $first_task  ${others[2]} ${others[3]} ${others[1]} $layer $hidden $try
      run_configuration $first_task  ${others[2]} ${others[1]} ${others[3]} $layer $hidden $try
      run_configuration $first_task  ${others[3]} ${others[2]} ${others[1]} $layer $hidden $try
      run_configuration $first_task  ${others[1]} ${others[2]} ${others[3]} $layer $hidden $try
done
done
done
