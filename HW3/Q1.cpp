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
  start_time = MPI_Wtime(); 
  if(rank == 0) {
    strcpy(message, "Hello, world");
    for (i=1; i<size; i++)
      MPI_Send(message,13,MPI_CHAR,i,type,MPI_COMM_WORLD);
  } else {
      MPI_Recv(message,20,MPI_CHAR,0,type,MPI_COMM_WORLD,&status);
    }
    printf( "%.13s from langer.k %d\n", message,rank);
    MPI_Finalize();
    end_time = MPI_Wtime();

    if (rank == 0) { 
        printf("Took %f\n", end_time-start_time);
    } 
}
  
