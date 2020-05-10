#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <mpi.h>


void zero_init_matrix(double ** matrix, size_t N);
void header(FILE *file);
void print_info(FILE *file, double start, double end, long unsigned int len, int n);
char* concatinate(char *s1, char *s2);


int main(int argc, char ** argv)
{
	int n = 0;
	int phlag = 0;
	int eps = 25;
	FILE *file;
	file = fopen("Table.txt", "a");

	char *str = argv[1];

	for(int i = 0; i < atoi(argv[2]); i++)
	{
		for (int j = 0; j < 3; j++)
		{
			str = concatinate(str, argv[1]); // enlargement of initial string, which can be set up in arguments
		}
	}

	//printf("%s\n", str); // - do not uncomment this until you want to see gigantic strings in ouput files

	///////////////////////////////////////	
	int flag = 0;
	int send;
	int address;
	MPI_Status status;
	MPI_Init(&argc, &argv);
	//MPI_Init(NULL, NULL);
	int p_size, p_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &p_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p_size);	
	///////////////////////////////////////	
	double start = MPI_Wtime();
	double end;

	send = 0;

	//if (p_rank == 0)
	//{
		//header(file); // This was not the way to go, I didn't came up with solution of how to put header in file 1 time for such program, so I just created another program to put appropriate header
	//}

	for(;;)
	{
		if (p_rank != send)
		{
			while(!flag)
			{
				//MPI_probe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status); MPI_probe was advised, but again Iprobe works
				MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status);
			}

			
			//MPI_Irecv(str, 1 + strlen(str), MPI_CHAR, status.MPI_SOURCE, 0, MPI_COMM_WORLD, &request);
			MPI_Recv(str, 1 + strlen(str), MPI_CHAR, status.MPI_SOURCE, 0, MPI_COMM_WORLD, &status);
			//MPI_Irecv(&n, 1  MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD, &request);
			MPI_Recv(&n, 1, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			flag = 0;
			send = p_rank; // now it's this proc's turn to send
			n = n + 1; // increment number of "resendings"
			end = MPI_Wtime();
			
			if ((end - start) > eps)
			{
				print_info(file, start, end, strlen(str), n);
				fclose(file);
				MPI_Abort(MPI_COMM_WORLD, 0); //abort process to gather information
			}
		}
		else
		{
			while(!phlag) 
			{
				address = rand() % p_size; // choosing random proc out of whole pool without sending one
				if (address != send) phlag = 1;
			} 
			phlag = 0;

			MPI_Ssend(str, 1 + strlen(str), MPI_CHAR, address, 0, MPI_COMM_WORLD);
			//MPI_Send(str, 1 + strlen(str), MPI_CHAR, address, 0, MPI_COMM_WORLD); does not work well, so Ssend instead

			MPI_Send(&n, 1, MPI_INT, address, 1, MPI_COMM_WORLD);
           	send = address;

		}
	}
}

void zero_init_matrix(double ** matrix, size_t N)
{
}

void header(FILE *file)
{
	fprintf(file, "Size(bytes)\t# Iterations\tTotaltime(secs)\tTime per message\tBandwidth(MB/s)\n") ;
}

void print_info(FILE *file, double start, double end, long unsigned int len, int n)
{
	fprintf(file, "%lu\t%d\t%f\t%f\t%f\n", len, n, end-start, (double)(end-start)/(double)n, (double) n*len / (end-start) / (1024. * 1024.)) ;
}

char *concatinate(char *a, char *b) 
{	
	char *ptr =(char *) malloc(strlen(a) + strlen(b) + 1); 
    return strcat(strcpy(ptr, a), b);
}
// https://stackoverflow.com/questions/34053859/concatenate-3-strings-and-return-a-pointer-to-the-new-string-c
