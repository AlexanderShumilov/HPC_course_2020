#!/bin/bash
mpicc cell_mpi.c -o c_mpi -lm

s="Thetext,whichpurportstobefromMPI,asksthereceivertoclickonalinktogetarefundviaane-transfer."
#for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 
for (( i=4; i <= 200; i+=4 ))
do
	echo $i 
	mpiexec --oversubscribe -np 16 c_mpi $i>> Cell_time_16.txt
done
