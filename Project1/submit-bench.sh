#! /bin/sh 
#BSUB -J submit-bench
#BSUB -o output2.txt
#BSUB -e error_file2
#BSUB -n 16
#BSUB -q ser-par-10g 
#BSUB cwd ~/EECE5640/Project1
work=~/EECE5640/Project1
cd $work
./placer

