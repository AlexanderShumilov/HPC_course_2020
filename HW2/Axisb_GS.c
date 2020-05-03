#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

void zero_init_matrix(double ** matrix, size_t N);
void zero_init_array(double * array, size_t N);
void rand_init_array(double * array, size_t N);
void diag_dom_init_matrix(double ** matrix, size_t N);
void laplacian_matrix(double ** matrix, size_t N);
void is_diag_dom(double ** matrix, size_t N);
double ** malloc_matrix(size_t N);
void print_matrix(double ** matrix, size_t N);
void print_array(double * array, size_t N);
void free_matrix(double ** matrix, size_t N);
void Gauss_Seidel_seq(double ** matrix, double * b,  int N);
void Gauss_Seidel_par(double ** matrix, double * b,  int N);

int main()
{
    const size_t N = 100; // size of an matrix

    clock_t start, end;   
 
    double ** A; // matrices
	double * rhs;

	rhs = (double *)malloc(N * sizeof(double));

    printf("Starting:\n");

    A = malloc_matrix(N); 
 
	rand_init_array(rhs, N);
	//zero_init_array(rhs, N);
	//rhs[0] = 1;

	//print_array(rhs, N);
	//printf("\n");

    laplacian_matrix(A, N);
	
	//print_matrix(A, N);
	//printf("\n");

	is_diag_dom(A, N);
	printf("\n");
	Gauss_Seidel_seq(A, rhs, N);
	printf("\n");
	Gauss_Seidel_par(A, rhs, N);
		
	free_matrix(A, N);
	free(rhs);
}


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


void zero_init_array(double * array, size_t N)
{
    for (int i = 0; i < N; i++)
    {
        array[i] = 0.0;  
    }
}

void rand_init_array(double * array, size_t N)
{
    for (int i = 0; i < N; i++)
    {
        array[i] = random() / (double)RAND_MAX + 10;  
    }
}

void diag_dom_init_matrix(double ** matrix, size_t N)
{
    srand(time(NULL));

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrix[i][j] = random() / (double)RAND_MAX + 10;
			if (i == j)
				{
					matrix[i][j] += 1000;
				}
        }
    }
}

void laplacian_matrix(double ** matrix, size_t N)
{
    srand(time(NULL));
	matrix[0][0] = 4;
    for (int i = 1; i < N; i++)
    {
        for (int j = 1; j < N; j++)
        {
            matrix[i][j] = 0;
			if (i == j)
				{
					matrix[i][j] = 4;

					matrix[i][j - 1] = -1;
					matrix[i - 1][j] = -1;
					
				}
        }
    }


	for (int i = 4; i < N; i++)
    {
        for (int j = 0; j < N - 4; j++)
        {
            if (i - 4 == j)
				{
					matrix[i][j] = -1;
				}
        }
    }

	for (int i = 0; i < N - 4; i++)
    {
        for (int j = 4; j < N; j++)
        {
            if (i == j - 4)
				{
					matrix[i][j] = -1;
				}
        }
    }

}

void is_diag_dom(double ** matrix, size_t N)
{
	int flag = 1;
    for (int i = 0; i < N; i++)
    {
		int sum = 0;
        for (int j = 0; j < N; j++)
        {
            sum += abs(matrix[i][j]);
		}
		sum -= abs(matrix[i][i]); 
		if (abs(matrix[i][i]) < sum)
			{
				flag = 0;
			}
        }
	printf("Matrix is diagonally dominant: %d\n", flag);
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

void print_matrix(double ** matrix, size_t N)
{

for (int row=0; row<N; row++)
{
    for(int columns=0; columns<N; columns++)
    {
         printf("%f\t", matrix[row][columns]);
    }
    printf("\n");
}

}

void print_array(double * array, size_t N)
{

	for (int row=0; row<N; row++)
	{
		printf("%f\t", array[row]);
	}
	printf("\n");
}

void free_matrix(double ** matrix, size_t N)
{
    for (int i = 0; i < N; ++i)
    {   
        free(matrix[i]);
    }
    
    free(matrix);
}


void Gauss_Seidel_seq(double ** matrix, double * b,  int N)
{

	int max_iter = N;
    double eps = 1e-10;
	float norm, norm_sol;
	int flag = 0, k = 0, count;
	double *solution;
	double *previous;
	int iter = 0;
	double start, end;
	
	solution = (double *)malloc(N * sizeof(double));
	previous = (double *)malloc(N * sizeof(double));
	zero_init_array(solution, N);

	//////////////////start//////////////////
	start = omp_get_wtime();	

	while ((!flag) && (iter < 2*N*N))
	{

		iter++;
		for (int n = 0; n < N; n++) 
			{
				previous[n] = solution[n];
			}


    	for (int i = 0; i < N; i++)
		{
			double sigma = 0;

			for (int t = 0; t < i; t++)
			{
				sigma += (matrix[i][t] * solution[t]);
			}

      		for (int j = i + 1; j < N; j++)
			{
				sigma += matrix[i][j] * previous[j];
      		}

			solution[i] = 1.0 / matrix[i][i] * (b[i] - sigma); 
		}
    	count = 0;


		norm = 0.0;
		norm_sol = 0.0;

		for (int k = 0; k < N; k++) 
		{
			norm += (previous[k] - solution[k])*(previous[k] - solution[k]);

		}

		for (int k = 0; k < N; k++) 
		{
			norm_sol += (solution[k])*(solution[k]);

		}
				
		
		if (norm / norm_sol < eps*eps) 
		{
			printf("Converged in %d iterations\n", iter);
			flag = 1;
		}
		
		//print_array(solution, N);
		if (iter == 2*N*N)
		{
			printf("Number of iterations is exceeded");
			printf("\n");
		}

  	}

	end = omp_get_wtime();
	//////////////////end//////////////////
	printf("Time for sequential solver: %f", end - start);
	printf("\n");
}




void Gauss_Seidel_par(double ** matrix, double * b,  int N)
{

	int max_iter = N;
    double eps = 1e-10;
	float norm, norm_sol;
	int flag = 0, k = 0, count;
	double *solution;
	double *previous;
	int iter = 0;
	double start, end;

	int num = 4;
	int chunk = 30;
		
	solution = (double *)malloc(N * sizeof(double));
	previous = (double *)malloc(N * sizeof(double));
	zero_init_array(solution, N);

	//////////////////start//////////////////
	start = omp_get_wtime();	

	while ((!flag) && (iter < 2*N*N))
	{
		iter++;
		#pragma omp parallel for schedule(dynamic, chunk) num_threads(num)
		for (int n = 0; n < N; n++) 
			{
				previous[n] = solution[n];
			}

		#pragma omp parallel for schedule(dynamic, chunk) num_threads(num)
    	for (int i = 0; i < N; i++)
		{
			double sigma = 0;

      		for (int t = 0; t < i; t++)
			{

				sigma += (matrix[i][t] * solution[t]);
			}

      		for (int j = i + 1; j < N; j++)
			{
				sigma += matrix[i][j] * previous[j];
      		}

			solution[i] = 1.0 / matrix[i][i] * (b[i] - sigma); 
		}
    	count = 0;
		
		norm = 0.0;
		norm_sol = 0.0;

		#pragma omp parallel for reduction(+:norm) schedule(dynamic, chunk) num_threads(num)
		for (int k = 0; k < N; k++) 
		{
			norm += (previous[k] - solution[k])*(previous[k] - solution[k]);

		}

		#pragma omp parallel for reduction(+:norm_sol) schedule(dynamic, chunk) num_threads(num)
		for (int k = 0; k < N; k++) 
		{
			norm_sol += (solution[k])*(solution[k]);

		}
				

		if (norm / norm_sol < eps*eps) 
		{
			printf("Converged in %d iterations\n", iter);
			flag = 1;
		}
		
		//print_array(solution, N);
		if (iter == 2*N*N)
		{
			printf("Number of iterations is exceeded");
			printf("\n");
		}

  	}

	end = omp_get_wtime();
	//////////////////end//////////////////
	printf("Time for parallelized solver: %f", end - start);
	printf("\n");
}
