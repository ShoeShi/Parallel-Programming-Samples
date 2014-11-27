README 11.27.14
/****************************************************************************
Sample MPI and OpenMP programs
Author: JX
https://github.com/BurningKoy
****************************************************************************/

Requirements:
	OpenMP library (enable openmp on Visual Studio)
	MPI support

Test environment 
	Windows 7 64bit
	Visual Studio 2013 linked with Microsoft HPC Pack 2012
	Gnome Linux with MPI support

1MPIColumnWiseMat
	Computing Ab=X with Columnwise distribution
	- Minimum block size is a single column
	- N processes must be a power of 2.
	One arg: [numprocs] 
	
	On Linux:
	mpicc -Wall -g main.c -o out
	mpirun -n #processes  ./out

	Windows:
	Compile
	mpiexec #processes *.exe
	
	
2MPIHeatEquation
	- *Any number of processes greater than 3
	See comment block inside A2Q4.c for a long description.
	One arg: [numprocs] 
	
	On Linux:
	mpicc -Wall -g A2Q2.c -o q4
	mpirun -n 5 ./q4
	
	Windows:
	Compile
	mpiexec #processes *.exe
	
	
1OpenmpColumnWiseMat
	Computing Ab=X with Columnwise distribution
	Two args: [numthreads] [numElements] 
	
	
	Windows and linux:
	Compile with /openmp or /fopenmp
	