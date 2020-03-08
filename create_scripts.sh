#!/bin/bash

if [ -z "$1" ]; then
  echo "Empty arguments"
  exit 1
fi

prefix=$1
cp template_job.slurm ${1}_trec.slurm
cp template_job.slurm ${1}_sst.slurm
sed 's/first_task=(trec)/first_task=(sst)/g' -i ${1}_sst.slurm
cp template_job.slurm ${1}_cola.slurm
sed 's/first_task=(trec)/first_task=(sst)/g' -i ${1}_cola.slurm
cp template_job.slurm ${1}_subj.slurm
sed 's/first_task=(trec)/first_task=(sst)/g' -i ${1}_subj.slurm
