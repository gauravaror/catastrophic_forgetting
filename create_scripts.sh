#!/bin/bash

if [ -z "$1" ]; then
  echo "Empty arguments"
  exit 1
fi

run_prefix=$1

create_configs() 
{
	architecture=$1
	prefix=$2
	template_file=$3
	cp $template_file jb_${architecture}_${prefix}_trec.slurm
	cp $template_file jb_${architecture}_${prefix}_sst.slurm
	sed 's/first_task=(trec)/first_task=(sst)/g' -i jb_${architecture}_${prefix}_sst.slurm
	cp $template_file jb_${architecture}_${prefix}_cola.slurm
	sed 's/first_task=(trec)/first_task=(cola)/g' -i jb_${architecture}_${prefix}_cola.slurm
	cp $template_file jb_${architecture}_${prefix}_subj.slurm
	sed 's/first_task=(trec)/first_task=(subjectivity)/g' -i jb_${architecture}_${prefix}_subj.slurm
}

cp template_job.slurm template_job_mlp.slurm
create_configs m $run_prefix template_job_mlp.slurm

sed 's/mlp/lstm/'  template_job.slurm > template_job_lstm.slurm
create_configs l $run_prefix template_job_lstm.slurm

sed 's/mlp/cnn/'  template_job.slurm > template_job_cnn.slurm
create_configs c $run_prefix template_job_cnn.slurm

sed 's/mlp/transformer/'  template_job.slurm > template_job_transformer.slurm
create_configs t $run_prefix template_job_transformer.slurm

