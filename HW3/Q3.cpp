#include <stddef.h>
#include <stdlib.h>
#include "mpi.h"
#include <stdio.h>
#include <string.h>
main(int argc, char **argv ) {
    double start_time, end_time, exe_time; 
    start_time = MPI_Wtime(); 
    int token, NP, myrank;
    MPI_Status status;
    MPI_Init (&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &NP);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank); 
    if (myrank != 0) {
        MPI_Recv(&token, 1, MPI_INT, myrank - 1, 0, 
        MPI_COMM_WORLD, &status);
        printf("Process %d received token %d from process %d\n",myrank, token, myrank - 1);
    } else {
        token = -1;
    }
    token += 2;
    MPI_Send(&token, 1, MPI_INT, (myrank + 1) % NP, 0, MPI_COMM_WORLD);
    if (myrank == 0) {
        MPI_Recv(&token, 1, MPI_INT, NP - 1, 0, MPI_COMM_WORLD, &status);
        printf("Process %d received token %d from process %d\n", myrank, token, NP - 1);
        end_time = MPI_Wtime();
        if (myrank == 0) { 
            printf("Took %f\n", end_time-start_time);
        } 
    }
    MPI_Finalize();
}
  
