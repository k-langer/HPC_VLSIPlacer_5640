#! /bin/sh 
#BSUB -J submit-bench
#BSUB -o output.txt
#BSUB -e error_file
#BSUB -n 32
#BSUB -q ht-10g
#BSUB cwd ~/EECE5640/HW2
work=~/EECE5640/HW2
cd $work
./HW2 100000000 p

