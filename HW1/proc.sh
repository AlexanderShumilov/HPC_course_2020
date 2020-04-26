gcc -pthread HW_1.2.c -o proc.exe -lm
for p in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
do
	./proc.exe 0 1 1000000 $p >> out.txt
done

