#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int ** malloc_matrix(int N, int M);
void print_matrix(int ** matrix, int N, int M);
void print_to_file(int ** matrix, int N, int M, FILE *fp);
void print_square(int ** matrix, int N, int M);
void copy(int **destmat, int **srcmat, int N, int M);
int **generate(int **pos, int N, int M, int N_suspected, int N_infected);
int **find_ind(int **array, int N, int M, int center_i, int center_j, int radius);
int **infect(int **pos, int N, int M, int infection_time, int radius);
int **step_release_loop(int **pos, int N, int M, int infection_time, int removed_time);

int main()
{
    int **pos, **pos2;
    int N = 5, M = 8, radius = 2, infection_time = 3, removed_time = 5;
    int N_epochs = 80, N_suspected = M*N/2, N_infected = 8;
    FILE *f;
    f = fopen ("out.txt", "wb");
    
    pos = malloc_matrix(N, M);
    pos = generate(pos, N, M, N_suspected, N_infected);
    //print_matrix(pos, N, M);
    print_square(pos, N, M);
    print_to_file(pos, N, M, f);

    for(int i = 0; i < N_epochs; i++)
    {
        pos = infect(pos, N, M, radius, infection_time);
        printf("\n\n");
        pos = step_release_loop(pos, N, M, infection_time, removed_time);
        //print_matrix(pos, N, M);
        print_square(pos, N, M);
        print_to_file(pos, N, M, f);
    }
    fclose(f);
    return 0;
}


int ** malloc_matrix(int N, int M)
{
    int ** matrix = (int **)malloc(N * sizeof(int *));
    
    for (int i = 0; i < N; ++i)
    {   
        matrix[i] = (int *)malloc(M * sizeof(int));
    }
    
    return matrix;
}

void print_matrix(int ** matrix, int N, int M)
{

    for (int row=0; row<N; row++)
    {
        for(int columns=0; columns<M; columns++)
        {
             printf("%d\t", matrix[row][columns]);
        }
        printf("\n");
    }
}

void print_to_file(int ** matrix, int N, int M, FILE *fp)
{

    for (int row=0; row<N; row++)
    {
        for(int columns=0; columns<M; columns++)
        {
             fprintf(fp, "%d\t", matrix[row][columns]);
        }
        fprintf(fp, "\n");
    }
    //fprintf(fp, "\n");
}

void print_square(int ** matrix, int N, int M)
{

    for (int row=0; row<N; row++)
    {
        for(int columns=0; columns<M; columns++)
        {
            if(matrix[row][columns] >= 2)
		    {
                printf("\u25A0 ");
	    	}        
		    else
		    {
                if(matrix[row][columns] == 1)
		        {
                    printf("\u25A1 ");
	    	    }
	    	    else
	    	    {
	    	        printf("%d ", 0);
	    	    }
		    }
        }
        printf("\n");
    }
}

void copy(int ** destmat, int ** srcmat, int N, int M) 
{
    //memcpy(destmat,srcmat, N*M*sizeof(int));
    for(int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            destmat[i][j] = srcmat[i][j];
        }
    }
    
}

int ** generate(int **pos, int N, int M, int N_suspected, int N_infected)
{
    for(int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            pos[i][j] = 0;
        }
    }
    
    for(int i = 0; i < N_suspected; i++)
    {
        int i_N = rand() % N;
        int j_M = rand() % M;
        if (pos[i_N][j_M] == 0)
        {
            pos[i_N][j_M] = 1;
        }
        else
        {
            i -= 1;
        }
    }
    
    for(int i = 0; i < N_infected; i++)
    {
        int i_N = rand() % N;
        int j_M = rand() % M;
        if (pos[i_N][j_M] == 0)
        {
            pos[i_N][j_M] = 2;
        }
        else
        {
            i -= 1;
        }
    }
    
    return pos;
}

int **find_ind(int **array, int N, int M, int center_i, int center_j, int radius)
{
    double seed;
    int **norms;
    srand(time(NULL));
    norms = malloc_matrix(N, M);
    
    for(int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            if ((((i - center_i)*(i - center_i) + (j - center_j)*(j - center_j)) <= radius*radius) && (array[i][j] == 1))
            {
                seed = rand()/(double)RAND_MAX;
                if (seed < 0.8)
                    {
                    array[i][j] = 2;
                    }
            }
        }
    }

    return array;
}

int **infect(int **pos, int N, int M, int infection_time, int radius)
{
    int **pos_init;
    pos_init = malloc_matrix(N, M);
    
    copy(pos_init, pos, N, M);

    for(int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            if ((pos_init[i][j] > 1))// && (pos_init[i][j] < 2 + infection_time))
            {
                pos = find_ind(pos, N, M, i, j, 1);
            }
        }
    }
    
    return pos;
}


int **step_release_loop(int **pos, int N, int M, int infection_time, int removed_time)
{
    int **pos_init;
    pos_init = malloc_matrix(N, M);
    double seed;
    
    copy(pos_init, pos, N, M);

    for(int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            if (pos_init[i][j] >= 2)
            {
                pos[i][j] += 1;
            }
            

            if (pos_init[i][j] >= 2 + infection_time + removed_time)
            {
                seed = rand()/(double)RAND_MAX;
                if (seed < 0.3)
                {
                    pos[i][j] = 0;
                }
                else
                {
                    pos[i][j] = 1;
                }
            }
        }
    }
    
    return pos;
}
