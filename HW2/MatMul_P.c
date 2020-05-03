#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

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

    double start, end;   
 
    double ** A, ** B, ** C; // matrices
	
    printf("Starting:\n");

    A = malloc_matrix(N);
    B = malloc_matrix(N);
    C = malloc_matrix(N);    

    rand_init_matrix(A, N);
    rand_init_matrix(B, N);
    zero_init_matrix(C, N);

    start = omp_get_wtime();
	const int num = 2;	
	int chunk = 5;

	int i, j, k;
	#pragma omp parallel for shared(A, B, C) private(i, j, k) collapse(2) schedule(dynamic, chunk) num_threads(num)
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
	
    end = omp_get_wtime();

    printf("Time elapsed (ijn): %f seconds.\n", (double)(end - start));




	rand_init_matrix(A, N);
    rand_init_matrix(B, N);
    zero_init_matrix(C, N);

    start = omp_get_wtime();

	//omp_set_num_threads(omp_get_num_procs());
	#pragma omp parallel for shared(A, B, C) private(i, j, k) collapse(2) schedule(dynamic, chunk) num_threads(num)
	for(j = 0; j < N; j++)
		{
		for(i = 0; i < N; i++)
		{
			double dot = 0;
			for(k = 0; k < N; k++)
			{
				dot += A[i][k] * B[k][j];
			}
			C[i][j] = dot;
		}
	}
	
    end = omp_get_wtime();

    printf("Time elapsed (jin): %f seconds.\n", (double)(end - start));





	rand_init_matrix(A, N);
    rand_init_matrix(B, N);
    zero_init_matrix(C, N);

    start = omp_get_wtime();

	//omp_set_num_threads(omp_get_num_procs());
	#pragma omp parallel for shared(A, B, C) private(i, j, k) collapse(2) schedule(dynamic, chunk) num_threads(num)
	for(k = 0; k < N; k++)
		{
		for(i = 0; i < N; i++)
		{
			double dot = 0;
			for(j = 0; j < N; j++)
			{
				dot += A[i][k] * B[k][j];
			}
			C[i][j] = dot;
		}
	}
	
    end = omp_get_wtime();

    printf("Time elapsed (nij): %f seconds.\n", (double)(end - start));
    free_matrix(A, N);
    free_matrix(B, N);
    free_matrix(C, N);

    return 0;
}
