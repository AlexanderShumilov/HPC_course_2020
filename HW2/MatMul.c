#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void zero_init_matrix(double ** matrix, size_t N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrix[i][j] = 0.0;
        }
    }
}

void rand_init_matrix(double ** matrix, size_t N)
{
    srand(time(NULL));

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrix[i][j] = rand() / RAND_MAX;
        }
    }
}

double ** malloc_matrix(size_t N)
{
    double ** matrix = (double **)malloc(N * sizeof(double *));
    
    for (int i = 0; i < N; ++i)
    {   
        matrix[i] = (double *)malloc(N * sizeof(double));
    }
    
    return matrix;
}

void free_matrix(double ** matrix, size_t N)
{
    for (int i = 0; i < N; ++i)
    {   
        free(matrix[i]);
    }
    
    free(matrix);
}

int main()
{
    const size_t N = 1000; // size of an array

    clock_t start, end;   
 
    double ** A, ** B, ** C; // matrices

    printf("Starting:\n");

    A = malloc_matrix(N);
    B = malloc_matrix(N);
    C = malloc_matrix(N);    

    rand_init_matrix(A, N);
    rand_init_matrix(B, N);
    zero_init_matrix(C, N);

    start = clock();

	int i, j, k;
	for(i = 0; i < N; i++)
		{
		for(j = 0; j < N; j++)
		{
			double dot = 0;
			for(k = 0; k < N; k++)
			{
				dot += A[i][k] * B[k][j];
			}
			C[i][j] = dot;
		}
	}

    end = clock();

    printf("Time elapsed (ijn): %f seconds.\n", (double)(end - start) / CLOCKS_PER_SEC);

    free_matrix(A, N);
    free_matrix(B, N);
    free_matrix(C, N);

    return 0;
}
