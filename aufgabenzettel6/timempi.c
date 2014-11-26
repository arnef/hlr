#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>
#include <mpi.h>

void get_host_timestamp(char *time_stamp) {
    char hostname[128];
    gethostname(hostname, sizeof hostname);

    struct timeval time;
    struct tm* ptm;
    char time_string[40];
    
    gettimeofday(&time, (struct timezone *) 0);
    ptm = localtime(&time.tv_sec);
    strftime(time_string, sizeof(time_string), "%Y-%m-%d %H:%M:%S", ptm);
    
    sprintf(time_stamp, "%s: %s.%d", hostname, time_string, (int) time.tv_usec);
}

int main (int argc, char** argv) {
    
    int rank;
    int size;
    
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    //get_host_timestamp();
    
    char message[170];
    MPI_Status status;
    int to;
    if (rank == 0) {
        int from;
        for (from = 1; from < size; from++) {
            MPI_Recv (message, 170, MPI_CHAR, from, 0, MPI_COMM_WORLD, &status);
            printf("%s\n", message);
        }
    }
    else {
        to = 0;
        get_host_timestamp(message);
        MPI_Send (message, strlen(message), MPI_CHAR, to, 0, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD); 
    printf("Rang %d beendet jetzt!\n", rank);
    MPI_Finalize();
    return 0;
}
