/****************************************************************************
Author: JX
Date: 10.21.14
https://github.com/BurningKoy

Computing Ab=X with Columnwise distribution
- BLOCKSIZE is unused, however blocksize is determined by N/num_procs
- Minimum block size is a single column

ii. Using num_procs as the block size is fine. 
(there should be 6 data points? not 5?).

****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include "assert.h"
#include "mpi.h"

#define NULLPTR 0x0
#define N 32
#define BLOCKSIZE 8

/*******************
The multidimensional array is just a 1D array, except there's a 
pointer to elements after every m.
*********************/
int **allocMatrix(int rows, int cols) 
{
	int i; 
	int *a = (int *)malloc(rows*cols*sizeof(int));
	int **a_rows = (int **)calloc(rows, sizeof(int *));
	for (i = 0; i < rows; i++)
		a_rows[i] = &(a[i*cols]); // point to columns

	assert(a_rows && a);
	return a_rows;
}


void freeMatrix(int **array) 
{
	free(array[0]);
	free(array);
	return;
}

int *allocArray(int size)
{
	int *b = (int *)malloc(size*sizeof(int));
	assert(b);
	return b;
}


void fillMatrix(int ** array, int numRows, int numCols)
{
	int i, j;
	srand((unsigned)time(NULL));
	for (i = 0; i < numRows; i++)
	{
		for (j = 0; j < numCols; j++)
			array[i][j] = (int)rand() % 100; //initialize with random ints between 0-99
	}

	return;
}

void fillArray(int * array, int size)
{
	int i;
	srand((unsigned)time(NULL));
	for (i = 0; i < size; i++)
			array[i] = (int)rand() % 10; //initialize with random ints between 0-9

	return;
}

void mpiRequestsFree(int *array, int size)
{
	int i;
	for (i = 0; i < size; i++)
		MPI_Request_free(&array[i]);
	free(array);
	return;
}


/*******************
numCols = numBlocks
numRows = N

returns array containing the A*b column.
And if there are multiple rows,  numBlocks of...A*b + A*b...
*********************/

int * colDotProd(int ** array, int * b, int numCols, int numRows)
{
	int i, j;
	int * result = (int*)calloc(numRows, sizeof(int));

	for (i = 0; i < numCols; i++)
	{
		for (j = 0; j < numRows; j++) //usually would need to determine numRows, but we know it's N.
		{
			result[j] += array[i][j] * b[i];
		}
	}

	return result;
}

/****************************************
Main program
*****************************************/

int main(int argc, char *argv[])
{
	int num_procs, my_rank, i, j, numBlocks;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

	double starttime = MPI_Wtime();
	numBlocks = N / num_procs; //assume no remainder

	// Ab = X
	//Create our Matrix A and Vector b and Vector X.
	int ** A = NULLPTR;
	int * b = NULLPTR, *X = NULLPTR, *sendbuf = NULLPTR;
	int * sendptr_A = NULLPTR;
	int * sendptr_b = NULLPTR;
	int ** recvbuf = allocMatrix(N, numBlocks);
	MPI_Datatype block_col, block_coltype;

	if (my_rank == 0)
	{// Split this up because other processes don't need to do this.
		X = allocArray(N);
		A = allocMatrix(N, N);
		sendptr_A = &(A[0][0]);
		fillMatrix(A, N, N);
	}

	//MAKE TYPE TO DESCRIBE AMOUNT TO SEND AND RECEIVE, required for columnwise distributing

	MPI_Type_vector(N, //Num columns
		1, //blocklength
		numBlocks, //Stride (Row size)
		MPI_INT, //Size of each element
		&block_col // Our Custom Column Type
		);
	MPI_Type_commit(&block_col);
	MPI_Type_create_resized(block_col, // old type
		0, // lower bound
		sizeof(int), //Size between each column
		&block_coltype // new type
		);
	MPI_Type_commit(&block_coltype);

	//Partition Matrix columwise
	MPI_Scatter(sendptr_A, numBlocks, block_coltype, &(recvbuf[0][0]), numBlocks, block_coltype, 0, MPI_COMM_WORLD);
	

	if (my_rank == 0)
	{

		//Send appropriate 'b' to each process.
		//Some pointer arithmatic here, since if we were to split 2 columns per process,
		//then next process would need to start at rank*numBlocks.
		b = allocArray(N);
		fillArray(b, N);
		sendptr_b = &(b[0]);
	}
	// this can replace the MPI_Scatter for b:
	//
	//	MPI_Request *sReqs = allocArray(num_procs);
	//	//Process 0 does a send to itself, because it saves some code.
	//	for (i = 0; i < num_procs; i++)
	//		MPI_Isend(&b[i*numBlocks], numBlocks, MPI_INT, i, my_rank, MPI_COMM_WORLD, &sReqs[i]);
	//	mpiRequestsFree(sReqs, num_procs);
	//}
	//
	////Receive b for appropriate blocks
	//int* b_block = allocArray(numBlocks);
	//MPI_Recv(b_block, numBlocks, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	//Partition b
	int* b_block = allocArray(numBlocks);
	MPI_Scatter(sendptr_b, numBlocks, MPI_INT, &(b_block[0]), numBlocks, MPI_INT, 0, MPI_COMM_WORLD);

	//Calculations on Columns
	sendbuf = colDotProd(recvbuf, b_block, numBlocks, N);
	MPI_Reduce(sendbuf, X, N, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	
	double endtime = MPI_Wtime();

	printf("\n\nProcess [%d]'s All recvbuf (raw data):", my_rank);
	for (i = 0; i < numBlocks; i++)
	{
		printf("\nColumn %d: ", i+1);
		for (j = 0; j < N; j++)
		{
			printf("%d ", recvbuf[i][j]);
		}
	}

	printf("\nProcess [%d]'s All sendbuf (calculated sum of local columns):", my_rank);
	for (i = 0; i < N; i++)
	{
		if (i % 10 == 0)
			printf("\n\t");
		printf("%d ", sendbuf[i]);
	}

	if (my_rank == 0)
	{
		printf("\n\nProcess [%d]'s Result Vector X:", my_rank);
		for (i = 0; i < N; i++)
		{
			if (i % 10 == 0)
				printf("\n\t");
			printf("%d ", X[i]);
		}
	}

	printf("\n\t[%d] (TIME) This process took %0.9f seconds.\n", my_rank, (endtime - starttime));

	if (my_rank == 0)
	{
		free(b);
		freeMatrix(A);
	}
	free(X);
	free(sendbuf);
	freeMatrix(recvbuf);
	MPI_Finalize();
	return(0);
}