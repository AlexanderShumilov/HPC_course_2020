#include<stdio.h>
#include<stdlib.h>
#include<stdbool.h>
#include<stdio.h>
#include<string.h>
#include<math.h>
#include<time.h>
//#include<mpi.h>

char* generate_pattern(int count);
int generate_rule();
char* generate_pattern_110(int count);
char* generate_pattern_43(int count);
void get_binary(char* rule_binary, int rule);
int get_index(int a, int b, int c);
void draw(int width, int* element);

int main(int argc, char* argv[])
{
    int width = atoi(argv[2]);
	printf("%d", width);
    char* init_pattern = malloc(width * sizeof(char));
	int prev;
    int next;
    int rule;
    int iterations = width;

	int* next_state = malloc(width * sizeof(int));
	int* element = malloc(width * sizeof(int));
	char* rule_binary = malloc(9 * sizeof(char));


	switch(atoi(argv[1]))
	{
		case 110:
		{
			rule = 110;
			init_pattern = generate_pattern_110(width);
		}
			break;

		case 184:
		{
			rule = 184;
			init_pattern = generate_pattern(width); // Here i pick random initial conditions because for this rule all of them give well-structured pictures
		}
			break;
	
		case 43:
		{
			rule = 43;
			init_pattern = generate_pattern_43(width);
		}
			break;

		default:
		{
			rule = generate_rule();
			//rule = atoi(argv[1]);
			init_pattern = generate_pattern(width);
		}
			break;
	}


	get_binary(rule_binary, rule);

	printf("\nRule: %d\n", rule);
	printf("Rule(binary): %s\n", rule_binary);
	printf("Number of cells (width of picture): %d\n", width);
	printf("Initial condition: %s\n", init_pattern);
	printf("Number of iterations (height of picture): %d\n\n", iterations);
	

    for(int j = 0; j < iterations; j++)
    {
//////////////////////////////////////////////////////

		if (j == 0)
		{
			for(int i = 0; i < width; i++)
    		{
    			if(init_pattern[i] == '0')
    			{
        			element[i] = 0;
					printf("\u25A1 ");
        		}
        		else 
					if(init_pattern[i] == '1')
            		{
            			element[i] = 1;
						printf("\u25A0 ");
            		}
    		}
			printf("\n");
		}

        for(int i = 0; i < width; i++)
    	{
        	if (i == 0)
			{
            	prev = width - 1;
			}
        	else
			{
            	prev = i - 1;
			}

        	if (i == (width - 1))
			{
            	next = 0;
			}
        	else
			{
            	next = i + 1;
			}

            next_state[i] = rule_binary[(int)get_index(element[next] % 2, element[i] % 2, element[prev] % 2)];
    	}

    	// copy next state to current elements
    	for (int k = 0; k < width; k++)
		{
			element[k] = next_state[k];
		}

		// Constant BC
		next_state[0] = element[0];
		next_state[width - 1] = element[width - 1];
        
		// Periodic BC
		//next_state[0] = element[width - 1];
		//next_state[1] = element[0];

		draw(width, element);
    }
//////////////////////////////////////////////////////
	printf("\n\n");
}



char* generate_pattern(int count)
{
	char* initial_condition = malloc(count * sizeof(char));

	srand(time(NULL));

	for(int i = 0; i < count; i++)
	{
		if (rand() % 2) 
		{
			initial_condition = strcat(initial_condition, "0");
		}
		else
		{
			initial_condition = strcat(initial_condition, "1");
		}
		//printf("%s ", initial_condition);
	}
	
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

void get_binary(char* rule_binary, int rule)
{
    for(int p = 0; p <= 7; p++)
    {
        if((int)(pow(2, p)) & rule)
        {
            rule_binary[abs(p - 7)] = '1';
        }
        else
        {
            rule_binary[abs(p - 7)] = '0';
        }
    }
    rule_binary[8] = '\0';
}

int get_index(int a, int b, int c)
{
	return(7 - a * pow(2, 0) - b * pow(2, 1) - c * pow(2, 2));
}


void draw(int width, int* element)
{

    for(int i = 0; i < width; i++)
    {
        if(element[i] == '1')
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

