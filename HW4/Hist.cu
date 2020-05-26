#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include <math.h>
//#define size 128

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__global__ void Calculate(int height, int width,  uint8_t *im, unsigned int *hist_bin, int N);
void Print(FILE *f, unsigned int *h, int N);

int main(int argc, char **argv)
{
	int N = 128;
	int width, height, comp;
	int i, j, ch, s;
	int n_channels = 3; // rgb

	uint8_t* image_non_scaled = stbi_load("fractal.jpg", &width, &height, &comp, 3);

	int block = N, grid = height * width / N;
	dim3 Block(block);
    	dim3 Grid(grid);
	uint8_t *im;
	unsigned int *hist_bin;
	unsigned int *hist = (unsigned int *) malloc(sizeof(unsigned int) * 256);

	FILE *f;
	f = fopen("Hist.txt", "wb");
	
	cudaMalloc(&im, sizeof(uint8_t) * height * width);
        cudaMalloc(&hist_bin, sizeof(unsigned int) * N);
        cudaMemset(hist_bin, 0, sizeof(unsigned int) * N);	
	uint8_t* image_scaled = (uint8_t *) malloc(sizeof(uint8_t) * height * width);

    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            s = 0;
            for (ch = 0; ch < n_channels; ch++)
            {
                s += image_non_scaled[i*width*n_channels + j*n_channels + ch];
            }
            image_scaled[i*width + j] = s / 3;
        }
    }

	cudaMemcpy(im, image_scaled, sizeof(uint8_t) * height * width, cudaMemcpyHostToDevice);
	Calculate<<<Grid, Block>>>(height, width, im, hist_bin, N);
	cudaDeviceSynchronize();	
	cudaMemcpy(hist, hist_bin, sizeof(unsigned int) * N, cudaMemcpyDeviceToHost);
	Print(f, hist, N);
	free(image_scaled);
	free(hist);
	cudaFree(im);
	cudaFree(hist_bin);
	fclose(f);

}

__global__ void Calculate(int height, int width,  uint8_t *im, unsigned int *hist_bin, int N)
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
	atomicAdd(&hist_bin[im[myId] % N], 1);
	
}

void Print(FILE *f, unsigned int *h, int N)
{
	for (int i = 0; i < N; i++)
	{
		fprintf(f, "%d\t", h[i]);
	}
	fprintf(f, "\n");
}
