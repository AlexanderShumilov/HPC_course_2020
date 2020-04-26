//command line arguments: 1 - leftmost point
//*********************** 2 - rightmost point
//*********************** 3 - total number of points
//*********************** 4 - total number of threads

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <semaphore.h>
#include <sys/time.h>

//function to get the timings
unsigned long get_time()
{
        struct timeval tv;
        gettimeofday(&tv, NULL);
        unsigned long ret = tv.tv_usec;
        ret /= 1000;
        ret += (tv.tv_sec * 1000);
        return ret;
}


//variables to store results of numerical integration
double mutex_res = 0;          // result of calculation using mutex method
double semaphore_res = 0;      // result of calculation using semaphore method
double busy_wait_res = 0;      // result of calculation using busy wait method

// global "flags"
pthread_mutex_t mutex;
sem_t semaphore;
int busy_wait_flag = 0;

// you may need this global variables, but you can make them inside main()
double a;                 // left point
double b;                 // right point
int n;                     // number of discretization points
double h;                 // distance between neighboring discretization points
int TOTAL_THREADS;
int n_per_thread = 0;

int appedix = 0;

//mathematical function that we calculate the arc length (declaration, define it yourselves)
double function(double x);

//function to calculate numerical derivative
double numerical_derivative(double x, int i);

//arc_length on a single thread
double serial_arc_length();

//multithreaded arc_length rule using busy waiting
void* busy_wait_arc_length(void*);
void busy_wait_main();

//multithreaded arc_length using mutex
void* mutex_arc_length(void*);
void mutex_main();

//multrthreaded arc_length using semaphore
void* semaphore_arc_length(void*);
void semaphore_main();

int main( int argc, char *argv[] )
{
    a = atoi(argv[1]);
    b = atoi(argv[2]);
    n = atof(argv[3]);
    h = (b-a)/n;
    TOTAL_THREADS = atoi(argv[4]);
	n_per_thread = n / TOTAL_THREADS;
    printf("TOTAL NUMBER OF THREADS: %d\n", TOTAL_THREADS);    
    long start = get_time();
    double duration;
    double result = serial_arc_length(0, n);
    duration = (get_time() - start);
    printf("Solution on a single thread: %f, time: %f ms\n", result, duration);

	double residue = n % TOTAL_THREADS;

    if(residue != 0)
    {
        appedix = n - TOTAL_THREADS * (n / TOTAL_THREADS);
    }
    busy_wait_main();    
    mutex_main();
    semaphore_main();
    return 0;
}

double function(double x)
{
    return cos(x)*exp(x);
}

double numerical_derivative(double x, int index)
{
	return (function(x + (index - 1) * h) - function(x + (index + 1) * h)) / 2 / h;
}

double serial_arc_length(int x_left, int N)
{
    double sum = 0;
    int x_right = 0;
	double part_1 = 0.0;
	double part_2 = 0.0;
	
    if( x_left == (TOTAL_THREADS-1) && (appedix != 0))
    {
    	x_right = appedix + N * (x_left + 1);
    }
    else
    {
        x_right = N * (x_left + 1);
    }
	// here goes integral discretization
    for(int i = N * x_left; i < x_right; i++)
    {
		sum += h * sqrt(1 + pow((function(a + (i - 1) * h) - function(a + (i + 1) * h)) / 2 / h, 2));	
    }

    return sum;
}


void* busy_wait_arc_length(void* process_rank)
{
    int rank = *((int *)process_rank);
    double sum = 0.0;

    sum = serial_arc_length(rank, n_per_thread);

    while(busy_wait_flag != rank)
	{
		// wait
	}

    busy_wait_res += sum; //accumulate

    busy_wait_flag++; // increase 'flag' value

    return NULL;
}

void* mutex_arc_length(void* process_rank)
{
    int rank = *((int *)process_rank);
    double sum = 0.0;

    sum = serial_arc_length(rank, n_per_thread);

    pthread_mutex_lock(&mutex); // locking

    mutex_res += sum; //accumulate

    pthread_mutex_unlock(&mutex); // unlocking

    return NULL;
}

void* semaphore_arc_length(void *process_rank)
{
    int rank = *((int *)process_rank);
    double sum = 0.0;

    sum = serial_arc_length(rank, n_per_thread);
    sem_wait(&semaphore);
    semaphore_res += sum;
    sem_post(&semaphore);

    return NULL;
}

// general constraction for this "mains" is the same (as far as I understand)

void busy_wait_main()
{
    pthread_t* thread_ptr;
    thread_ptr = malloc(TOTAL_THREADS * sizeof(pthread_t));

	int* point;
	point = malloc(TOTAL_THREADS * sizeof(int));

    long start = get_time();
    double duration;
	
	// start for pthread
	int i = 0;
    for(int i=0; i < TOTAL_THREADS; i++)
    {
		*(point + i) = i; 
        pthread_create(&(*(thread_ptr + i)), NULL, &busy_wait_arc_length, &(*(point + i)));
    }

    for(int i=0; i < TOTAL_THREADS; i++)
    {
        pthread_join(*(thread_ptr + i), NULL);
    }

    duration = (get_time() - start);
    printf("Solution using Busy Waiting: %f, time: %f ms\n", busy_wait_res, duration);

    free(thread_ptr); // finish for pthread - cleaning
}

void mutex_main()
{
	pthread_t* thread_ptr;
    thread_ptr = malloc(TOTAL_THREADS * sizeof(pthread_t));

    int* point;
	point = malloc(TOTAL_THREADS * sizeof(int));

    int i = 0;

    long start = get_time();
    double duration;

    pthread_mutex_init(&mutex, NULL); // start for mutex

    for(i=0; i < TOTAL_THREADS; i++)
    {
        *(point + i) = i; 
        pthread_create(&(*(thread_ptr + i)), NULL, &mutex_arc_length, &(*(point + i)));
    }

    for(i=0; i < TOTAL_THREADS; i++)
    {
        pthread_join(*(thread_ptr + i), NULL);
    }

    duration = (get_time() - start);
    printf("Solution using Mutex: %f, time: %f ms\n", mutex_res, duration);

    pthread_mutex_destroy(&mutex); // finish for mutex - cleaning
}

void semaphore_main()
{
    pthread_t* thread_ptr;
    thread_ptr = malloc(TOTAL_THREADS * sizeof(pthread_t));

    int* point;
	point = malloc(TOTAL_THREADS * sizeof(int));

	int i = 0;

    long start = get_time();
    double duration;

    sem_init(&semaphore, 0, 1); // start for semaphore
	
    for(i=0; i < TOTAL_THREADS; i++)
    {
        *(point + i) = i; 
        pthread_create(&(*(thread_ptr + i)), NULL, &semaphore_arc_length, &(*(point + i)));
    }

    for(i=0; i < TOTAL_THREADS; i++)
    {
        pthread_join(*(thread_ptr + i), NULL);
    }


    duration = (get_time() - start);
    printf("Solution using Semaphore: %f, time: %f ms\n", semaphore_res, duration);

    sem_destroy(&semaphore); // finish for semaphore - cleaning
}
