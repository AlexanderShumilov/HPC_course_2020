#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include <math.h>
//#define size 128

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__global__ void Median(int height, int width, int n_channels, double *hist, double *hist_bin);
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

	uint8_t* image_non_filtered = stbi_load("fractal.jpg", &width, &height, &comp, n_channels);
	
	//int block = N, grid = height * width / N;
	int block = width, grid = height * n_channels;
	dim3 Block(block);
    	dim3 Grid(grid);

	double *im = (double *) malloc(sizeof(double) * height * width * n_channels);
	double *im_median = (double *)malloc(sizeof(double) * height * width * n_channels);
	double *hist, *hist_bin;

	cudaMalloc(&hist, sizeof(double) * height * width * n_channels);
	cudaMalloc(&hist_bin, sizeof(double) * height * width * n_channels);

	/*
	unsigned int *hist = (unsigned int *) malloc(sizeof(unsigned int) * 256);

	FILE *f;
	f = fopen("Hist.txt", "wb");
	
	cudaMalloc(&im, sizeof(uint8_t) * height * width);
        cudaMalloc(&hist_bin, sizeof(unsigned int) * N);
	*/

	image_to_array(height, width, n_channels, im, image_non_filtered);
	cudaMemcpy(hist, im, sizeof(double) * height * width * n_channels, cudaMemcpyHostToDevice);
	Median<<<Grid, Block>>>(height, width, n_channels, hist, hist_bin);
	cudaDeviceSynchronize();	
	cudaMemcpy(im_median, hist_bin, sizeof(double) * height * width * n_channels, cudaMemcpyDeviceToHost);
	array_to_image(height, width, n_channels, im_median, image_non_filtered);
	//stbi_write_jpg("fractal_median.jpg", width, height, n_channels, image_non_filtered, width * n_channels);
	stbi_write_png("fractal_median.png", width, height, n_channels, image_non_filtered, width * n_channels);

	free(image_non_filtered);
	free(im);
	free(im_median);
	cudaFree(hist);
	cudaFree(hist_bin);
	
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

__global__ void Median(int height, int width, int n_channels, double *hist, double *hist_bin)
{
        int myId = threadIdx.x + blockDim.x * blockIdx.x;
	int stensils = 5, N = 120;
	int j = myId / width / n_channels;
	int i = myId / n_channels - j * width;
	int edge = myId - i * n_channels - j * width * n_channels; 	
	//double *window = (double*)malloc(sizeof(double)*(N + 1));
	double window[121];

		// boundary
		if (i < stensils - 1)
		{
			hist_bin[j*width*n_channels + i*n_channels + edge] =  hist[j*width*n_channels + i*n_channels + edge];
		}
		if (j < stensils - 1)
		{
			hist_bin[j*width*n_channels + i*n_channels + edge] =  hist[j*width*n_channels + i*n_channels + edge];
		}
		if (i > width - stensils)
		{
			hist_bin[j*width*n_channels + i*n_channels + edge] =  hist[j*width*n_channels + i*n_channels + edge];
		}
		if (j > height - stensils)
		{
			hist_bin[j*width*n_channels + i*n_channels + edge] =  hist[j*width*n_channels + i*n_channels + edge];
		}
		// not boundary
		if (i > stensils - 1 || j > stensils - 1 || i < width - stensils || j < height - stensils)
		{
			
			int idx = 0;
			for (int x = - stensils; x <= stensils; x++)
			{
				for (int y = - stensils; y <= stensils; y++)
				{
					window[idx] = hist[(j + y)*width*n_channels + (i + x)*n_channels + edge];
					idx += 1;	
				}
			}
			//sort entries in window
			for (int x = 0; x < N; x++)
			{
				for (int y = 0; y < N - x + 1; y++)
				{
					double tmp = 0;
        				if (window[y] > window[x])
        				{
                				tmp = window[x];
                				window[x] = window[y];
                				window[y] = tmp;
       					 }

				}
			}
			//sort(window, N + 1);

			hist_bin[j*width*n_channels + i*n_channels + edge] = window[N / 2];
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
