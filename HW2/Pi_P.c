#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

int main()
{
    int N = 100000000;
	int count = 0;

    double x, y;
    double begin, end;

	begin = omp_get_wtime();
    #pragma omp parallel private(x,y) num_threads(4)
    {
		unsigned int seed = omp_get_thread_num() + 1;
        #pragma omp for reduction(+: count) 
        for(int i = 0; i < N; i++)
        {
            x = (double)rand_r(&seed)/RAND_MAX;
            y = (double)rand_r(&seed)/RAND_MAX;
            if(x*x + y*y <= 1)
                count += 1;
        }
    }
    end = omp_get_wtime();

	printf("pi = %f\n", 4*(double)count/N);
    printf("Time : %f\n\n", end - begin);

	return 0;
}
