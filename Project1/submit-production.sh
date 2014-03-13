#! /bin/sh 
#BSUB -J submit-bench
#BSUB -o output3.txt
#BSUB -e error_file3
#BSUB -n 8
#BSUB -q ser-par-10g 
#BSUB cwd ~/EECE5640/Project1
work=~/EECE5640/Project1
cd $work
./placer

