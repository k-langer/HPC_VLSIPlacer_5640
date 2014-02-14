#! /bin/sh 
#BSUB -J Q1-HW3-Langer
#BSUB -o output.txt
#BSUB -e error.txt
#BSUB -n 1
#BSUB -q ser-par-10g
#BSUB cwd ~/EECE5640/Project1/
work=~/EECE5640/Project1/
cd $work

./placer

#watch output.txt
