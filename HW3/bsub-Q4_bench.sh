#! /bin/sh 
#BSUB -J Q4-HW3-Langer
#BSUB -o logs/Q4-20.txt
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

mpirun -np 20 -prot -TCP -lsf ./Q4 100
mpirun -np 20 -prot -TCP -lsf ./Q4 1000
mpirun -np 20 -prot -TCP -lsf ./Q4 10000
mpirun -np 20 -prot -TCP -lsf ./Q4 100000
mpirun -np 20 -prot -TCP -lsf ./Q4 1000000
mpirun -np 20 -prot -TCP -lsf ./Q4 10000000
mpirun -np 20 -prot -TCP -lsf ./Q4 100000000
mpirun -np 20 -prot -TCP -lsf ./Q4 1000000000
#watch output.txt
