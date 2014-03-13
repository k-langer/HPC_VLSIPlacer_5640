#! /bin/sh 
#BSUB -J submit-bench
#BSUB -o output.txt
#BSUB -e error_file
#BSUB -n 16
#BSUB -q ser-par-10g 
#BSUB cwd ~/EECE5640/Project1
work=~/EECE5640/Project1
cd $work
./placer

