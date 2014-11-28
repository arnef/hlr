#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>
#include <limits.h>
#include <mpi.h>


int main (int argc, char** argv) {
    
    int rank;
    int size;
    int minimum[1];
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    char message[45];
    MPI_Status status;
        char hostname[5];
        gethostname(hostname, sizeof hostname);

        struct timeval time;
        struct tm* ptm;
        char time_string[40];
    
        gettimeofday(&time, (struct timezone *) 0);
        ptm = localtime(&time.tv_sec);
        strftime(time_string, sizeof(time_string), "%Y-%m-%d %H:%M:%S", ptm);
        int microsecond[1];
        if (rank == 0) {
            microsecond[0] = INT_MAX;
        }
        else {
            microsecond[0] = (int) time.tv_usec; 
        }
        MPI_Reduce(microsecond, minimum, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    int to;
    if (rank == 0) {
        int from;
        for (from = 1; from < size; from++) {
            MPI_Recv (message, 45, MPI_CHAR, from, 0, MPI_COMM_WORLD, &status);
            printf("%s\n", message);
        }
        printf("%d\n", minimum[0]);
    }
    else {
        to = 0;
        sprintf(message, "%s: %s.%d", hostname, time_string, (int) time.tv_usec);
        MPI_Send (message, 45, MPI_CHAR, to, 0, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD); 
    printf("Rang %d beendet jetzt!\n", rank);
    
    MPI_Finalize();
    return 0;
}
