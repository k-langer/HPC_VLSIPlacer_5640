#! /bin/sh 
#BSUB -J Q4-HW3-Langer
#BSUB -o logs/output.txt
#BSUB -e logs/error_file
#BSUB -n 20
#BSUB -q ht-10g
#BSUB cwd ~/EECE5640/HW3
work=~/EECE5640/HW3
cd $work
mkdir -p logs 

tempfile1=.hostlistrun 
tempfile2=.hostlist-tcp 

echo $LSB_MCPU_HOSTS > $tempfile1 
declare -a hosts 
read -a hosts < ${tempfile1} 
for ((i=0; i<${#hosts[@]}; i += 2)) ; 
 do 
 HOST=${hosts[$i]} 
 CORE=${hosts[(($i+1))]} 
 echo $HOST:$CORE >> $tempfile2 
done 

mpirun -np 20 -prot -TCP -lsf ./Q4 $1

#watch output.txt
