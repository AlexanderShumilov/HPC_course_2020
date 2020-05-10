#!/bin/bash
rm ./Table.txt
gcc header.c -o head
./head
mpicc Ping_pong.c -o pp -lm

s="Thetext,whichpurportstobefromMPI,asksthereceivertoclickonalinktogetarefundviaane-transfer."
#for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 
for (( i=1; i <= 99; i++ ))
do
	echo "Current iteration: $i"
	mpiexec --oversubscribe -np 4 pp $s $i >> log.txt
done
