#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

int main()
{

    srand(0);

    int N = 100000000;
	int count = 0;

    double x, y;
    double begin, end;

	begin = omp_get_wtime();

    for(int i = 0; i < N; i++)
        {
        x = (double)random()/RAND_MAX;
        y = (double)random()/RAND_MAX;
        if(x*x + y*y <= 1)
            count += 1;
        }

    end = omp_get_wtime();

	printf("pi (sequential) = %f\n", 4*(double)count/N);
    printf("Time : %f\n\n", end - begin);

	return 0;
}
