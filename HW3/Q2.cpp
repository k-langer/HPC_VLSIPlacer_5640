#include <stddef.h>
#include <stdlib.h>
#include "mpi.h"
#include <stdio.h>
#include <string.h>
main(int argc, char **argv ) {
    char message[20];
    int i,rank, size, type=99;
    double start_time, end_time, exe_time; 
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Barrier(MPI_COMM_WORLD);   
    strcpy(message, "Hello, world");
    start_time = MPI_Wtime(); 
    MPI_Bcast(message,13, MPI_CHAR, 0,MPI_COMM_WORLD);
    printf( "%.13s from langer.k %d\n", message,rank);
    end_time = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) { 
        printf("Took %f\n", end_time-start_time);
    } 
    MPI_Finalize();
}
  
