#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include <math.h>

void Print_matrix(int N, FILE *f, double *m, long int n);
__global__ void Jacobi(long int n, double *in, double *out);

int main(void)
{
	int N = 128;
	int block = 1024;
        int grid = N * N / 1024;
	double *heat = (double *) calloc(sizeof(double), N * N), *arr, *arr_out;
	cudaMalloc(&arr, sizeof(double) * N * N);
        cudaMalloc(&arr_out, sizeof(double) * N * N);
        cudaMemcpy(arr, heat, sizeof(double) * N * N, cudaMemcpyHostToDevice);
 
	dim3 Block(block);
        dim3 Grid(grid);

        int k_iter = 0;

	for (int i = 0; i < N; i++)
	{
		heat[N * i] = 1;
	}
		
	FILE *f = fopen("heat.txt", "wb");

	for(;;)
	{
		k_iter++;
		Jacobi<<<Grid, Block>>>(N, arr, arr_out);
		cudaMemcpy(heat, arr, sizeof(double)*N*N, cudaMemcpyDeviceToHost);
		Print_matrix(N, f, heat, N);
		if (k_iter >= 200)
		{
			break;
		}		
	}
	
	cudaDeviceSynchronize();
	fclose(f);
	free(heat);
	cudaFree(arr);
	cudaFree(arr_out);
}

__global__ void Jacobi(long int n, double *in, double *out)
{

	int myId, i, j;
	int north, south, east, west;
	//double N, S, E, W;
	int index_center;
	//int flag = 0;
    	myId = threadIdx.x + blockDim.x * blockIdx.x;
	i = myId / n;
	j = myId - n * i;

	index_center = i*n + j;

	south = j - 1 > 0 ? (j - 1) + i*n : 0;
	west = i - 1 > 0 ? j + (i - 1)*n : -1;

	north = j + 1 < n - 1 ? (j + 1) + i*n : -1;
	east = i + 1 < n - 1 ? j + (i + 1)*n : -1;
	//in[0] = 0;
	//if (j < 1) {S = 0;} else {S = in[(int)south];}
	//if (j > n - 2) {N = 0;} else {N = in[(int)north];}
	//if (i > n - 2) {E = 0;} else {E = in[(int)east];}
	//if (i < 1) {W = 0;} else {W = in[(int)west];}
//	if (i < n - 1 && j < n - 1 && i>0 && j>0)
//	{
		out[index_center] = 0.25 * (in[(int)north] + 
					    in[(int)south] + 
					    in[(int)east] + 
					    in[(int)west]);
			
	/*	out[index_center] = 0.25 * (S+ 
					    N + 
					    E + 
					    W);
	*/	
//	}
  	//__syncthreads();
	if (i == 0)
		{
			out[index_center] = 0;
		}

	if (i == n - 1)
		{
			out[index_center] = 0;
		}

	if (j == n - 1)
		{
			out[index_center] = 0;
		}

//	__syncthreads();

	if (j == 0) 
		{
			out[index_center - j] = 1;
		}		

  	__syncthreads();
	in[index_center] = out[index_center];      
	
}

void Print_matrix(int N, FILE *f, double *m, long int n)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			fprintf(f, "%f\t", m[i*N + j]);
		}
	}
	fprintf(f, "\n");
}

