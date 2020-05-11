#include<stdio.h>
#include <mpi.h>
#include<stdlib.h>
#include<stdbool.h>
#include<stdio.h>
#include<string.h>
#include<math.h>
#include<time.h>

int* generate_pattern(int count);
int generate_rule();
char* generate_pattern_110(int count);
char* generate_pattern_43(int count);
void get_binary(int* rule_binary, int rule);
int get_index(int a, int b, int c);
void draw(int width, int* element);
void print_array(int* array, int i);

int main(int argc, char* argv[])
{
    int width = atoi(argv[1]);
	//int width = 12;
	double start;
	int *array = (int *)calloc(sizeof(int), width);
    char* init_pattern = malloc(width * sizeof(char));
	int prev;
    int next;
    char neighbourhood[4];
    int rule;
    int iterations = width;

	int* next_state = malloc(width * sizeof(int));
	int* element = malloc(width * sizeof(int));
	int* rule_binary = malloc(8 * sizeof(int));

///////////////////////////////
	int p_size;
    int p_rank;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &p_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p_size);
	int boundary_left = width / p_size * p_rank;
	int boundary_right =  width / p_size * (p_rank + 1);
	int *size_array = (int *)calloc(sizeof(int), p_size);
	int *num = (int *)calloc(sizeof(int), p_size);
///////////////////////////////
	rule = generate_rule();
	
	if (p_rank == 0)
	{
		//printf("\nRule: %d\n", rule);
		//get_binary(rule_binary, rule);
		//printf("Rule(binary): ");
		//print_array(rule_binary, width);
		//printf("\n");
		//printf("Number of cells (width of picture): %d\n", width);
		//printf("Number of iterations (height of picture): %d\n\n", iterations);
		start = MPI_Wtime();
	}

	get_binary(rule_binary, rule);
	for (int k = 0; k < p_size; k++)
	{
		size_array[k] = (int) width / (double) p_size;
		num[k] = k * size_array[k];
	}

	size_array[p_size - 1] = size_array[p_size - 1] + width % p_size;

	if (p_rank == (p_size - 1))
	{
		boundary_right = width; 
	}

	int chunk = boundary_right - boundary_left;	
	chunk += 2; // 2 ghost cells (in general to the left and to the right of current cell)
	int* par_elem = (int *)calloc(sizeof(int), chunk);
	int* next_par_elem = (int *)malloc(sizeof(int) * chunk);
	int* sender = (int *)malloc(sizeof(int) * (chunk - 2));

	par_elem = generate_pattern(chunk);

	
    for(int j = 0; j < iterations; j++)
    {
		//print_array(par_elem, chunk);
//////////////////////////////////////////////////////
		MPI_Barrier(MPI_COMM_WORLD);
//////////////////////////////////////////////////////
		MPI_Send(&par_elem[1], 1, MPI_INT, (p_rank - 1) < 0 ? p_size - 1 : p_rank - 1, 0, MPI_COMM_WORLD);
		MPI_Send(&par_elem[chunk - 2], 1, MPI_INT, (p_rank + 1) % p_size, 1, MPI_COMM_WORLD);
//////////////////////////////////////////////////////
		MPI_Recv(&par_elem[chunk - 1], 1, MPI_INT, (p_rank + 1) % p_size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&par_elem[0], 1, MPI_INT, (p_rank - 1) < 0 ? p_size - 1 : p_rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		
///////////////////////////////
        for(int i = 1; i < chunk; i++)
    	{
        	if (i == 0)
			{
            	prev = chunk - 1;
			}
        	else
			{
            	prev = i - 1;
			}

        	if (i == (chunk - 1))
			{
            	next = 0;
			}
        	else
			{
            	next = i + 1;
			}

            next_par_elem[i] = rule_binary[get_index(par_elem[next], par_elem[i], par_elem[prev])];
    	}
		//print_array(next_par_elem, chunk);
    	// copy next state to current
    	for (int k = 1; k < chunk; k++)
		{
			par_elem[k] = next_par_elem[k];

		}

		// Constant BC
		par_elem[0] = par_elem[0];
		par_elem[chunk - 1] = par_elem[chunk - 1];
        
/////////////////////////////////////////
		if (j % 1 == 0)
		{
			for (int l = 1; l < (chunk - 1); l++) 
			{
				sender[l - 1] = par_elem[l];
			}
			MPI_Barrier(MPI_COMM_WORLD);	
			MPI_Gatherv(sender, chunk - 2, MPI_INT, array, size_array, num, MPI_INT, 0, MPI_COMM_WORLD);
		}

    }
//////////////////////////////////////////////////////
	if (p_rank == 0)
	{
		free(array);
		printf(" %f\n", MPI_Wtime() - start);
	}

	free(par_elem);
	free(rule_binary);

	MPI_Finalize();
	
}



int* generate_pattern(int count)
{
	int* initial_condition = malloc(count * sizeof(int));
	
	initial_condition[0] = 1;
	for(int i = 0; i < count; i++)
	{
		if (rand() % 2) 
		{
			initial_condition[i] = 0;
		}
		else
		{
			initial_condition[i] = 1;
		}
	}
	//print_array(initial_condition, count);
	//printf("\n");
	return(initial_condition);
}

char* generate_pattern_110(int count)
{
	char* initial_condition = malloc(count * sizeof(char));

	for(int i = 0; i < count - 1; i++)
	{
		if (rand() % 2) 
		{
			initial_condition = strcat(initial_condition, "0");
		}
		else
		{
			initial_condition = strcat(initial_condition, "0");
		}
		//printf("%s ", initial_condition);
	}
	initial_condition = strcat(initial_condition, "1");
	return(initial_condition);
}


char* generate_pattern_43(int count)
{
	char* initial_condition = malloc(count * sizeof(char));
	
	initial_condition = strcat(initial_condition, "1");
	for(int i = 1; i < count - 1; i++)
	{
		if (rand() % 2) 
		{
			initial_condition = strcat(initial_condition, "0");
		}
		else
		{
			initial_condition = strcat(initial_condition, "0");
		}
		//printf("%s ", initial_condition);
	}
	initial_condition = strcat(initial_condition, "1");
	return(initial_condition);
}

int generate_rule()
{
	srand(time(NULL));

	return(rand() % 256);
}

void get_binary(int* rule_binary, int rule)
{
    for(int p = 0; p <= 7; p++)
    {
        if((int)(pow(2, p)) & rule)
        {
            rule_binary[abs(p - 7)] = 1;
			//printf("%d", rule_binary[p]);
        }
        else
        {
            rule_binary[abs(p - 7)] = 0;
			//printf("%d", rule_binary[p]);
        }
    }
	//printf("\n");
}

int get_index(int a, int b, int c)
{
	return(7 - a * pow(2, 0) - b * pow(2, 1) - c * pow(2, 2));
}


void draw(int width, int* element)
{

    for(int i = 0; i < width; i++)
    {
        if(element[i] == 1)
		{
            printf("\u25A0 ");
		}        
		else
		{
            printf("\u25A1 ");
		}
    }

    printf("\n");
}

void print_array(int* array, int n)
{
	for(int i = 0; i < n; i++)
	{
		printf("%d", array[i]);
	}
}
