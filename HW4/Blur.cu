#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include <math.h>
//#define size 128

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__global__ void Blur(int height, int width, int n_channels, double *kernel, double *hist, double *hist_bin);
//void Print(FILE *f, unsigned int *h, int N);
void image_to_array(int height, int width, int n_channels, double *im, uint8_t *image_non_filtered);
void array_to_image(int height, int width, int n_channels, double *im, uint8_t *image_non_filtered);
void swap(int x, int y, double *window);

int main(int argc, char **argv)
{
	int N = 128;
	int width, height, comp;
	int i, j, ch, s;
	int n_channels = 3; // rgb
	int dim_kernel = 5 * 5;

	uint8_t* image_non_filtered = stbi_load("fractal.jpg", &width, &height, &comp, n_channels);
	
	//int block = N, grid = height * width / N;
	int block = width, grid = height * n_channels;
	dim3 Block(block);
    	dim3 Grid(grid);

	double *im = (double *) malloc(sizeof(double) * height * width * n_channels);
	double *im_median = (double *)malloc(sizeof(double) * height * width * n_channels);
	double *hist, *hist_bin, *cuda_kernel;

	cudaMalloc(&hist, sizeof(double) * height * width * n_channels);
	cudaMalloc(&hist_bin, sizeof(double) * height * width * n_channels);
	cudaMalloc(&cuda_kernel, sizeof(double)*9);

	double *box_kernel = (double *) calloc(sizeof(double), dim_kernel);
	for(i = 0; i < dim_kernel; i++)
	{
		box_kernel[i] = 1 / (double)dim_kernel;
	}

	double *gaussian_kernel = (double *) calloc(sizeof(double), dim_kernel);
	gaussian_kernel[0] = gaussian_kernel[4] = gaussian_kernel[20] = gaussian_kernel[24] = 1 / 256.;
	gaussian_kernel[1] = gaussian_kernel[3] = gaussian_kernel[5] = gaussian_kernel[15] = gaussian_kernel[9] = gaussian_kernel[19] = gaussian_kernel[21] = gaussian_kernel[23] = 4 / 256.;
	gaussian_kernel[2] = gaussian_kernel[10] = gaussian_kernel[14] = gaussian_kernel[22] = 6 / 256.;
	gaussian_kernel[6] = gaussian_kernel[8] = gaussian_kernel[16] = gaussian_kernel[18] = 16 / 256.;
	gaussian_kernel[7] = gaussian_kernel[11] = gaussian_kernel[13] = gaussian_kernel[17] = 24 / 256.;
	gaussian_kernel[12] = 36 / 256.;

	/*
	unsigned int *hist = (unsigned int *) malloc(sizeof(unsigned int) * 256);

	FILE *f;
	f = fopen("Hist.txt", "wb");
	
	cudaMalloc(&im, sizeof(uint8_t) * height * width);
        cudaMalloc(&hist_bin, sizeof(unsigned int) * N);
	*/
	//cudaMemcpy(cuda_kernel, gaussian_kernel, sizeof(double) * dim_kernel, cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_kernel, box_kernel, sizeof(double) * dim_kernel, cudaMemcpyHostToDevice);	
	
	image_to_array(height, width, n_channels, im, image_non_filtered);
	cudaMemcpy(hist, im, sizeof(double) * height * width * n_channels, cudaMemcpyHostToDevice);
	Blur<<<Grid, Block>>>(height, width, n_channels, cuda_kernel, hist, hist_bin);
	cudaDeviceSynchronize();	
	cudaMemcpy(im_median, hist_bin, sizeof(double) * height * width * n_channels, cudaMemcpyDeviceToHost);
	array_to_image(height, width, n_channels, im_median, image_non_filtered);
	//stbi_write_jpg("fractal_filtered.jpg", width, height, n_channels, image_non_filtered, width * n_channels);
	stbi_write_png("fractal_filtered.png", width, height, n_channels, image_non_filtered, width * n_channels);

	free(image_non_filtered);
	free(im);
	free(im_median);
	free(box_kernel);
	free(gaussian_kernel);
	cudaFree(hist);
	cudaFree(hist_bin);
	cudaFree(cuda_kernel);
}

void image_to_array(int height, int width, int n_channels, double *im, uint8_t *image_non_filtered)
{
for (int i = 0; i < height * width * n_channels; i++)
	{
		im[i] = (double)image_non_filtered[i];
	}
}
void array_to_image(int height, int width, int n_channels, double *im, uint8_t *image_non_filtered)
{
for (int i = 0; i < height * width * n_channels; i++)
	{
		image_non_filtered[i] = uint8_t(im[i]);
	}
}

void swap(int x, int y, double *window)
{
	double tmp = 0;
	if (window[y] > window[x])
	{
		tmp = window[x];
		window[x] = window[y];
		window[y] = tmp;
	}
}
/*
void swap(int *xp, int *yp)  
{  
    int temp = *xp;  
    *xp = *yp;  
    *yp = temp;  
}  
  
// A function to implement bubble sort  
void bubbleSort(double* arr, int n)  
{  
    int i, j;  
    for (i = 0; i < n-1; i++)      
      
    // Last i elements are already in place  
    for (j = 0; j < n-i-1; j++)  
        if (arr[j] > arr[j+1])  
            swap(&arr[j], &arr[j+1]);  
} 
*/

__global__ void Blur(int height, int width, int n_channels, double *kernel, double *hist, double *hist_bin)
{
        int myId = threadIdx.x + blockDim.x * blockIdx.x;
	int i = myId / width / n_channels;
	int j = myId / n_channels - i * width;
	int edge = myId - j * n_channels - i * width * n_channels; 	
	//double *window = (double*)malloc(sizeof(double)*(N + 1));

		// boundary
		if (i < 1)
		{
			hist_bin[i*width*n_channels + j*n_channels + edge] =  hist[i*width*n_channels + j*n_channels + edge];
		}
		if (j < 1)
		{
			hist_bin[i*width*n_channels + j*n_channels + edge] =  hist[i*width*n_channels + j*n_channels + edge];
		}
		if (i > width - 2)
		{
			hist_bin[i*width*n_channels + j*n_channels + edge] =  hist[i*width*n_channels + j*n_channels + edge];
		}
		if (j > height - 2)
		{
			hist_bin[i*width*n_channels + j*n_channels + edge] =  hist[i*width*n_channels + j*n_channels + edge];
		}
		// not boundary
		if (i >= 1 || j >= 1 || i <= width - 2 || j <= height - 2)
		{
		//for(int k = 0; k < 5; k++)
		//{
			
			int pos = 0;
			for(int x = -2; x <= 2; x++)
				{
					for(int y = -2; y <= 2; y++)
					{
						hist_bin[i * width * n_channels + j * n_channels + edge] = hist[(i+x) * width * n_channels + (j+y) * n_channels + edge] * kernel[pos];
						pos += 1;
					}	
				}
		//}
		}
}

/*
void Print(FILE *f, unsigned int *h, int N)
{
	for (int i = 0; i < N; i++)
	{
		fprintf(f, "%d\t", h[i]);
	}
	fprintf(f, "\n");
}
*/
