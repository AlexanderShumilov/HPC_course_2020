#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cstring>
#include <omp.h>
#include <time.h>
#define RGB_COMPONENT_COLOR 255

struct PPMPixel {
    int red;
    int green;
    int blue;
};

typedef struct{
    int x, y, all;
    PPMPixel * data;
} PPMImage;


void readPPM(const char *filename, PPMImage& img);
void writePPM(const char *filename, PPMImage & img);
void sequential(PPMImage &img, PPMImage &image);
void parallel(PPMImage &img, PPMImage &image);
void fill_colors(double ** A, double ** B, double ** C, PPMImage &img, PPMImage &image);
void reverse(double ** A, double ** B, double ** C, PPMImage &image);
char *concatenate(const char *a, const char *b, const char *c);



int main()
{
    
    int N = 3000;
	int sym_max = 50;
    char *name = (char *)malloc(sym_max * sizeof(char));
    char f_name[6];
    double start, end;

	PPMImage car, shifted;

    readPPM("car2.ppm", car);

    start = omp_get_wtime();

    for (int i = 0; i < N; i++)
	{
		sequential(car,shifted);

		car = shifted;

		sprintf(f_name, "%d", i);
		if ((i % 500) == 0)
		{
			name = concatenate("car_", (const char *) f_name, ".ppm");
			writePPM((const char *)name, shifted);
		}
    }

    end = omp_get_wtime();

    printf("\nTime of sequential calculations %f\n", end - start);
    delete(car.data);  
    free(name);



	readPPM("car2.ppm", car);

    start = omp_get_wtime();

    for (int i = 0; i < N; i++)
	{
		parallel(car,shifted);

		car = shifted;

		sprintf(f_name, "%d", i);
		if ((i % 500) == 0)
		{
			name = concatenate("car_", (const char *) f_name, ".ppm");
			writePPM((const char *)name, shifted);
		}
    }

    end = omp_get_wtime();

    printf("\nTime of parallel calculations %f\n", end - start);
    delete(car.data);  
    free(name);


    return 0;
}


void readPPM(const char *filename, PPMImage& img){
    std::ifstream file (filename);
    if (file){
        std::string s;
        int rgb_comp_color;
        file >> s;
        if (s!="P3") {std::cout<< "error in format"<<std::endl; exit(9);}
        file >> img.x >>img.y;
        file >>rgb_comp_color;
        img.all = img.x*img.y;
        std::cout << s << std::endl;
        std::cout << "x=" << img.x << " y=" << img.y << " all=" <<img.all;
        img.data = new PPMPixel[img.all];
        for (int i=0; i<img.all; i++){
            file >> img.data[i].red >>img.data[i].green >> img.data[i].blue;
        }

    }else{
        std::cout << "the file:" << filename << "was not found" << std::endl;
    }
    file.close();
}

void writePPM(const char *filename, PPMImage & img){
    std::ofstream file (filename, std::ofstream::out);
    file << "P3"<<std::endl;
    file << img.x << " " << img.y << " "<< std::endl;
    file << RGB_COMPONENT_COLOR << std::endl;

    for(int i=0; i<img.all; i++){
        file << img.data[i].red << " " << img.data[i].green << " " << img.data[i].blue << (((i+1)%img.x ==0)? "\n" : " ");
    }
    file.close();
}

void fill_colors(double ** A, double ** B, double ** C, PPMImage &img, PPMImage &image)
{

	int n = 0;

	for (int j = 0; j < image.x; j++)
	{
		A[j] = (double *)malloc(image.y * sizeof(double));  
		B[j] = (double *)malloc(image.y * sizeof(double));
		C[j] = (double *)malloc(image.y * sizeof(double)); 
	}

	for (int j = 0; j < image.y; j++)
	{
		for (int i = 0; i < image.x; i++)
		{
			A[i][j] = img.data[n].red;
			B[i][j] = img.data[n].blue;
			C[i][j] = img.data[n].green;

			n++;
		}
	}
}


void reverse(double ** A, double ** B, double ** C, PPMImage &image)
{

	int n = 0;

	for (int j = 0; j < image.y; j++)
	{
		for (int i = 0; i < image.x; i++)
		{
			image.data[n].red = A[i][j];
            image.data[n].blue = B[i][j];
            image.data[n].green = C[i][j];

            n++;
		}
	}
}

void sequential(PPMImage &img, PPMImage &image){
	
	image.x = img.x;
	image.y = img.y;
	image.all = image.x * image.y;

	image.data = new PPMPixel[image.all];

	double ** A = (double **)malloc(sizeof(double *)*image.x);
	double ** B = (double **)malloc(sizeof(double *)*image.x);
	double ** C = (double **)malloc(sizeof(double *)*image.x);


	fill_colors(A, B, C, img, image);	

	for (int i = 0; i < image.x; i++)
	{
		for (int j = 0; j < image.y; j++)
		{
			A[i][j] = A[(i + 1) % image.x][j];
			B[i][j] = B[(i + 1) % image.x][j];
			C[i][j] = C[(i + 1) % image.x][j];
		}
	}
	
	reverse(A, B, C, image);
	
	
}



void parallel(PPMImage &img, PPMImage &image){
	
	image.x = img.x;
	image.y = img.y;
	image.all = image.x * image.y;
	int num = 2;

	image.data = new PPMPixel[image.all];

	double ** A = (double **)malloc(sizeof(double *)*image.x);
	double ** B = (double **)malloc(sizeof(double *)*image.x);
	double ** C = (double **)malloc(sizeof(double *)*image.x);

	fill_colors(A, B, C, img, image);
	
	#pragma omp parallel for schedule(dynamic, 10), num_threads(num)
	for (int i = 0; i < image.x; i++)
	{
		for (int j = 0; j < image.y; j++)
		{
			A[i][j] = A[(i + 1) % image.x][j];
			B[i][j] = B[(i + 1) % image.x][j];
			C[i][j] = C[(i + 1) % image.x][j];
		}
	}	

	reverse(A, B, C, image);
	
}


char *concatenate(const char *a, const char *b, const char *c) 
{	
	char *ptr =(char *) malloc(strlen(a) + strlen(b) + strlen(c) + 1); 
    return strcat(strcat(strcpy(ptr, a), b), c);
}
// https://stackoverflow.com/questions/34053859/concatenate-3-strings-and-return-a-pointer-to-the-new-string-c
