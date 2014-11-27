/****************************************************************************
Author: JX
date : 11.25.14
https://github.com/BurningKoy

Columnwise matrix calculation using OpenMP - SPMD, Parallel for, and Seq

Uncomment the printArray to validate the vectors 
obtained from the different approaches.

Results:
(tested for T = 16, N = 1048576)
SPMD Crit. Reg. is always slower than Seq.
SPMD Red. execution time faster than Seq. for large N
Parallel for is always faster than Seq. and SPMD Red.
Calloc may be the cause of some margin of error.

Set a high N for a more visible speedup.
* *************************************************************************** */


#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <limits.h>
#include <math.h>
#include <time.h>

//#define NUM_THREADS  -- Moved this to Main as an argument.
static int N;  // global

/****************************************
Print to file
*****************************************/
void printToFile(int nx, int ny, int **u, char *filename)
{
	int i, j;
	FILE *fp;

	fp = fopen(filename, "w");
	for (i = 0; i < nx; i++)
	{
		for (j = 0; j < ny; j++)
		{
			fprintf(fp, "%d", u[i][j]);
			if (j != ny - 1)
				fprintf(fp, " ");
			else
				fprintf(fp, "\n");
		}
	}
	fclose(fp);

	return;
}


void printArray(int * b, int n)
{
	int i;
	for (i = 0; i < n; i++)
	{
	printf("%d", b[i]);
			if ( i != n - 1)
				printf(" ");
	}

	return;
}


void intToFile(int n, char *filename)
{
	FILE *fp = fopen(filename, "a");
	if (fp != NULL)
		fprintf(fp, "[%d]\n", n);
	fclose(fp);
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
}


void fillArray(int * array, int size)
{
	int i;
	srand((unsigned)time(NULL));
	for (i = 0; i < size; i++)
		array[i] = (int)rand() % 7; //initialize with random ints between 0-99
}


int **allocMatrix(int rows, int cols)
{
	int i;
	int *a = (int *)malloc(rows*cols*sizeof(int));
	int **a_rows = (int **)calloc(rows, sizeof(int *));
	for (i = 0; i < rows; i++)
		a_rows[i] = &(a[i*cols]); // point to columns

	return a_rows;
}

void freeMatrix(int **array)
{
	free(array[0]);
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
			result[j] += array[i][j] * b[i];
	}
	return result;
}


/*******************
Ax = b
returns b.
*********************/
int * colWiseMultiply(int ** A, int * x, int nx, int ny){
	int i, j;
	int * b;

#pragma omp parallel private (i, j) shared(b)
	{
#pragma omp master
		b = (int *)calloc(ny, sizeof(int)); //initialize with 0s.

		for (j = 0; j < ny; j++){ //flipped the loop
#pragma omp for nowait
			for (i = 0; i < nx; i++){
#pragma omp critical
				{//make sure we dont read b[j] at the same time.
					b[i] += A[i][j] * x[i];
				}
			}
		}
	}

	return b;
}


int * colWiseMultiplyRed(int ** A, int * x, int nx, int ny){
	int i, j;
	int * b;
	b = (int *)calloc(ny, sizeof(int)); //initialize with 0s.

#pragma omp parallel private (i, j) shared(b)
	{
		for (j = 0; j < ny; j++){ //flipped the loop
#pragma omp for nowait reduction(+:b[i])
			for (i = 0; i < nx; i++){
					b[i] += A[i][j] * x[i];
			}
		}
	}

	return b;
}


int * colWiseMultiplyFor(int ** A, int * x, int nx, int ny){
	int i, j;
	int * b;
	b = (int *)calloc(ny, sizeof(int)); //initialize with 0s.

		for (j = 0; j < ny; j++){ //flipped the loop
#pragma omp parallel for nowait reduction(+:b[i])
			for (i = 0; i < nx; i++){
				b[i] += A[i][j] * x[i];
			}
		}
	

	return b;
}


int * colWiseMultiplySeq(int ** A, int * x, int nx, int ny){
	int i, j;
	int * b;

	b = (int *)calloc(ny, sizeof(int)); //initialize with 0s.

	for (j = 0; j < ny; j++){ //flipped the loop
		for (i = 0; i < nx; i++){
			{//make sure we dont read b[j] at the same time.
				b[i] += A[i][j] * x[i];
			}
		}
	}

	return b;
}

int main(int argc, char *argv[])
{
	double starttime, endtime;
	int i;

	if (argc != 3){
		printf("\nPlease set the number of threads(int) and num_elements(int)");
		exit(EXIT_SUCCESS);
	}

	printf("\nNumber of threads=%s", argv[1]);
	printf("\nNumber of elements=%s", argv[2]);
	omp_set_num_threads(atoi(argv[1]));
	N = (int)sqrt(strtod(argv[2], NULL));

	int ** A = allocMatrix(N, N);
	int * x = (int *)malloc(N*sizeof(int));
	int * b;
	fillMatrix(A, N, N);
	fillArray(x, N);

	printToFile(N, N, A, "A3Q2(Initial)MatrixA.txt");

	starttime = omp_get_wtime();
	b = colWiseMultiply(A, x, N, N);
	endtime = omp_get_wtime();
	printf("\n\t(TIME) This process(SPMD Crit. Reg.) took %0.9f seconds.\n", endtime - starttime);
	printf("\tFinal Vector b: ");
	//printArray(b, N);

	starttime = omp_get_wtime();
	b = colWiseMultiplyRed(A, x, N, N);
	endtime = omp_get_wtime();
	printf("\n\t(TIME) This process(SPMD Red.) took %0.9f seconds.\n", endtime - starttime);
	printf("\tFinal Vector b: ");
	printArray(b, N);


	starttime = omp_get_wtime();
	b = colWiseMultiplyFor(A, x, N, N);
	endtime = omp_get_wtime();
	printf("\n\t(TIME) This process(Parallel for) took %0.9f seconds.\n", endtime - starttime);
	printf("\tFinal Vector b: ");
	//printArray(b, N);

	//starttime = omp_get_wtime();
	b = colWiseMultiplySeq(A, x, N, N);
	//endtime = omp_get_wtime();
	//printf("\n\t(TIME) This process(Sequential) took %0.9f seconds.\n", endtime - starttime);
	printf("\tFinal Vector b: ");
	printArray(b, N);

	/*printf("\n[Exit Code]: %d. Press a button to exit.", EXIT_SUCCESS);*/
	//getchar();
	freeMatrix(A);
	exit(EXIT_SUCCESS);
}