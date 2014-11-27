/****************************************************************************
Author : JX
Date: 10.21.14
https://github.com/BurningKoy

Rod is length = 1
initData(), time = 0
According to our notes, the rod starts at 0 degrees at both ends,
and at x = 0.5, t is 100 degrees.

timeStep(), 0 < time <= 5
Model the cooling of the rod using the equation

Equation for temperature inside a point:
u_(i,j+1) = r*u_(i-1,j) + (1-2r)*u_(i,j) + r*u_(i+1,j) , where r = k/(h^2)

****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

#define MINPROCS	4
//This is my K = 0.001	H=0.25
//Parameters
#define NX			100          //5 seconds, time steps , number of rows
#define NY			20			//Length, number of columns
#define MAXTEMP		100

//Don't change these
#define PI			3.14159265
#define P_k			1.0/NX		//segments of time
//change segments of time 
#define	P_h			0.25		//segments on rod, size of one cell of NY sections. Errors if this becomse too small.
#define P_r			P_k/(P_h*P_h)

//Tags			   
#define START_TAG   -5                  
#define L_TAG       -4                  
#define R_TAG       -3                  
#define FIN_TAG     -2                 
#define NONEIGH	    -1   //No neighbor    

void init(float **u);
void update(int start, int end, int t, float **u);
float **allocMatrix(int rows, int cols);
void freeMatrix(float **u);
int *allocArray(int size);
//void pointerTests(int my_rank, float **u);
void printToFile(int nx, int ny, float **u, char *filename);
void debugToFile(int my_rank, char* msg, char *filename);
void intToFile(int my_rank, int step, char *filename);

/****************************************
Initialize Matrix
- Ends start at 0 degrees Celsius ( Ice bath )
- Center starts at 100 degrees Celsius
- Everywhere else is 100(sinPIEx)

*****************************************/
void init(float **u) 

{
	int i;
	for (i = 0; i < NY; i++)
		u[0][i] = (float)(MAXTEMP*sin(PI*((double)i / ((double)NY-1))));
	return;
}

/****************************************
Formula
start = process's column start
end = process's column end
*****************************************/
void update(int my_rank, int start, int end, int t, float ** u)
{
	int i, j;

	//No need to calculate the rod's ends.
	if (start == 0)
		start = 1;
	if (end == NY)
		end--;
	for (i = start; i < end; i++)
	{
		u[t][i] = (P_r * u[t - 1][i - 1])
			+ ((1 - 2 * P_r) * u[t - 1][i])
			+ (P_r * u[t - 1][i + 1]);

		//intToFile(my_rank, u[t - 1][i - 1], "calcs.txt");
		//intToFile(my_rank, u[t - 1][i], "calcs.txt");
		//intToFile(my_rank, u[t - 1][i + 1], "calcs.txt");
	}
	return;
}



/****************************************
Helpers
- Allocate/free
- Debugger output
- Pointer tests
*****************************************/
/****************************************
rows = NX
cols = NY

*****************************************/
//float
float ** allocMatrix(int rows, int cols)
{
	int i;
	float *a = (float *)malloc(rows*cols*sizeof(float));
	float **a_rows = (float **)calloc(rows, sizeof(float *));
	for (i = 0; i < rows; i++)
		a_rows[i] = &(a[i*cols]); // pointer to columns

	return a_rows;
}

void freeMatrix(float **array)
{
	free(array[0]);
	free(array);
	return;
}

//int
int *allocArray(int size)
{
	int *b = (int *)malloc(size*sizeof(int));
	return b;
}

//Pointer Tests
//void pointerTests(int my_rank, float ** u)
//{
//	intToFile(my_rank, u, "Array.txt");
//	intToFile(my_rank, &u[0], "Array.txt");
//	intToFile(my_rank, &u[0][0], "Array.txt");
//	intToFile(my_rank, *u, "Array.txt");
//	return;
//}

/****************************************
Print to file
*****************************************/
void printToFile(int nx, int ny, float **u, char *filename)
{
	int i, j;
	FILE *fp;

	fp = fopen(filename, "w");
	for (i = 0; i < nx; i++)
	{
		for (j = 0; j < ny; j++)
		{
			fprintf(fp, "%8.1f", u[i][j]);
			if (j != ny - 1)
				fprintf(fp, " ");
			else
				fprintf(fp, "\n");
		}
	}
	fclose(fp);

	return;
}

//Helps me see where I'm going
void debugToFile(int my_rank, char* msg, char *filename)
{
	FILE *fp = fopen(filename, "a");
	fprintf(fp, "[%d] %s: \n", my_rank, msg);
	fclose(fp);
	return;
}

void intToFile(int my_rank, int n, char *filename)
{
	FILE *fp = fopen(filename, "a");
	fprintf(fp, "[%d] %d \n", my_rank, n);
	fclose(fp);
	return;
}


//
//main
//

int main(int argc, char *argv[])
{
	int	my_rank, num_procs,
		mycols, avecols, offset, rmdr,
		left, right,
		i, j, t;


	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	MPI_Status status;
	MPI_Request sReqs;
	double starttime = MPI_Wtime();



	int * allColSizes = allocArray(num_procs); 
	int * allOffsets = allocArray(num_procs);
	float ** u = allocMatrix(NX, NY); //float alloc

	//Input 0s
	for (i = 0; i < NX; i++)
	for (j = 0; j < NY; j++)
		u[i][j] = 0.0;
	
	init(u);
	
	if( my_rank == 0 )
		printToFile(NX, NY, u, "initial.txt");
	//debugToFile(my_rank, "Initialized matrix.", "SlaveDebug.txt");

	if (num_procs < MINPROCS)
	{
		printf("Number of processes must be at or above %d.\n",
			MINPROCS);
		fflush(stdout);
		MPI_Abort(MPI_COMM_WORLD, 0);
		exit(1);
	}


	//
	//Partitioning
	//
	avecols = NY / num_procs; //my columns
	rmdr = NY % num_procs; //remainder
	offset = my_rank*avecols; //index of my columns
	
	//
	//Setup up for Allgatherv
	//
	//allOffsets[0] = 0;
	mycols = (my_rank < rmdr) ? avecols + 1 : avecols; // my columns
	for (i = 0; i < num_procs; i++)
	{
		allColSizes[i] = (i < rmdr) ? avecols + 1 : avecols;
		//if (i != num_procs-1)
		//allOffsets[i+1] = allOffsets[i] + allColSizes[i];
	}

	offset = 0;
	for (int i = 0; i < my_rank; i++)
		offset += allColSizes[i];

	MPI_Allgather(&offset, 1, MPI_INT, allOffsets, 1, MPI_INT, MPI_COMM_WORLD);
	//intToFile(my_rank, offset, "Offsets.txt");//debugging

	//Handle edge cases for workers at edge columns
	if (my_rank == 0)
		left = NONEIGH;
	else
		left = my_rank - 1;

	if (my_rank == num_procs - 1)
		right = NONEIGH;
	else
		right = my_rank + 1;

	printf("\n\t[%d] mycols= %d offset= %d ", my_rank, mycols, offset);
	printf("left= %d right= %d", left, right);

	// Do all 5 steps here
	// If statements handle the border requirement, ie when column # = 0 or numColumns
	// Send each neighboring process
	//	with your appropriate data as well as 
	// Receive neighboring process's data
	//
	// t is used as a tag.
	// Each process communicates with 2 others, except for processes at the edge.
	// 

	int left_offset = offset - 1;
	int right_offset = offset + mycols;
	float fbuffer;

	for (t = 1; t < NX; t++)
	{
		//debugToFile(my_rank, "Inside time step ", "SlaveDebug.txt");
		if (left != NONEIGH)
		{
			MPI_Isend(&u[t][offset], 1, MPI_FLOAT,
				left, t, MPI_COMM_WORLD, &sReqs); // I am from your right
			MPI_Recv(&fbuffer, 1, MPI_FLOAT,
				left, t, MPI_COMM_WORLD, &status); // here's their left

			if (status.MPI_SOURCE == right)
				u[t][right_offset] = fbuffer;
			else
				u[t][left_offset] = fbuffer;
		}

		if (right != NONEIGH)
		{
			MPI_Isend(&u[t][right_offset - 1], 1, MPI_FLOAT,
				right, t, MPI_COMM_WORLD, &sReqs); //I am from your left
			MPI_Recv(&fbuffer, 1, MPI_FLOAT,
				right, t, MPI_COMM_WORLD, &status); // I need this right value

			if (status.MPI_SOURCE == right)
				u[t][right_offset] = fbuffer;
			else
				u[t][left_offset] = fbuffer;
		}
		//New Calculations

		update(my_rank, offset, offset + mycols, t, u);

		//
		// Distribute results to all processors
		//
		MPI_Allgatherv(&(u[t][offset]), mycols, MPI_FLOAT,
			&(u[t][0]), allColSizes, allOffsets,
			MPI_FLOAT, MPI_COMM_WORLD);

	}
	
	double endtime = MPI_Wtime();
	if( my_rank == 0 )
		printToFile(NX, NY, u, "1finalpipetemp.txt");
	printf("\n\t[%d] (TIME) This process took %0.9f seconds.\n", my_rank, (endtime - starttime));

	MPI_Request_free(&sReqs);
	freeMatrix(u);
	free(allColSizes);
	free(allOffsets);

	MPI_Finalize();
	return 0;
}