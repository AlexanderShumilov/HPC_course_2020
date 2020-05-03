#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <time.h>

double ** malloc_matrix(size_t N);
void parametrization(double ** matrix, double *param, size_t N);
void print_matrix(double ** matrix, size_t N);
void free_matrix(double ** matrix, size_t N);
void grad_descent_sequential(double ** matrix, double lr, int steps, size_t N, double * param);
void grad_descent_parallel(double ** matrix, double lr, int steps, size_t N, double * param);
void relative_error(double * param, double * param_);

int main()
{
	int n = 1e+6;
	double lr = 1e-6;

	const size_t N = 2000; 
	double ** A;
	double param[2];

	param[0] = rand() / (double)RAND_MAX;
	param[1] = rand() / (double)RAND_MAX;	

	printf("System parameters to find: a = %f, b = %f\n", param[0], param[1]);
	printf("\n");

	A = malloc_matrix(N);
	parametrization(A, param, N);
	//print_matrix(A, N);
	grad_descent_sequential(A, lr, n, N, param);
	grad_descent_parallel(A, lr, n, N, param);
	


	free_matrix(A, N);
	return 0;
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



void parametrization(double ** matrix, double * param, size_t N)
{
	double noise;

	srand(time(NULL));

	for (int i = 0; i < N; i++)
    {	
		noise = rand() / (double)RAND_MAX;

    	matrix[i][0] = i * rand() / (double)RAND_MAX;
        matrix[i][1] = param[0] * matrix[i][0] + param[1] + noise;
    }
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

void free_matrix(double ** matrix, size_t N)
{
    for (int i = 0; i < N; ++i)
    {
        free(matrix[i]);
    }

    free(matrix);
}

void grad_descent_sequential(double ** matrix, double lr, int steps, size_t N, double * param)
{
	double param_[2];
	double start, end;
	double grad_a, grad_b;
	
	srand(time(NULL));

	param_[0] = rand() / (double)RAND_MAX;
	param_[1] = rand() / (double)RAND_MAX;
	
	start = omp_get_wtime();

	for (int s = 0; s < steps; s++)
	{
		grad_a = 0.;
		grad_b = 0.;
		
		for (int i = 0; i < N; i++)
		{
			grad_a += -2*(matrix[i][1] - param_[0]*matrix[i][0] - param_[1]) * matrix[i][0];
			grad_b += -2*(matrix[i][1] - param_[0]*matrix[i][0] - param_[1]);
		}
		
		param_[0] -= grad_a * lr;
		param_[1] -= grad_b * lr;

	}

	end = omp_get_wtime();

	printf("System parameters computed: a = %f, b = %f\n", param_[0], param_[1]);
	printf("Time: %f\n", end - start);
	printf("\n");
}

void grad_descent_parallel(double ** matrix, double lr, int steps, size_t N, double * param)
{
	double param_[2];
	double start, end;
	double grad_a, grad_b;

	int chunk = 20;
	int num = 4;
	srand(time(NULL));

	param_[0] = rand() / (double)RAND_MAX;
	param_[1] = rand() / (double)RAND_MAX;
	
	start = omp_get_wtime();

	for (int s = 0; s < steps; s++)
	{
		grad_a = 0.;
		grad_b = 0.;
		
		#pragma omp parallel for schedule(dynamic, chunk) reduction(+: grad_a) reduction(+: grad_b) num_threads(num)
		for (int i = 0; i < N; i++)
		{
			grad_a += -(matrix[i][1] - param_[0]*matrix[i][0] - param_[1]) * matrix[i][0] * 2;
			grad_b += -(matrix[i][1] - param_[0]*matrix[i][0] - param_[1]) * 2;
		}
		
		param_[0] -= grad_a * lr;
		param_[1] -= grad_b * lr;

	}

	end = omp_get_wtime();

	printf("System parameters computed: a = %f, b = %f\n", param_[0], param_[1]);
	printf("Time: %f\n", end - start);
	printf("\n");
	relative_error(param, param_);
}

void relative_error(double * param, double * param_)
{
	double norm = 0.0;
	double norm_sol = 0.0;
	int N = 2;

	for (int k = 0; k < N; k++) 
	{
	norm += (param[k] - param_[k])*(param[k] - param_[k]);

	}

	for (int k = 0; k < N; k++) 
	{
		norm_sol += (param[k])*(param[k]);

	}	

	printf("Relative error: %f\n", norm / norm_sol);

}

