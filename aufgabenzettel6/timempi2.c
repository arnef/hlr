#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>
#include <limits.h>
#include <mpi.h>


int main (int argc, char** argv) {
    
    // id des prozesses
    int rank;
    // anzahl der prozesse
    int size;
    int global_minimum_microseconds[1];  
    int microseconds[1];

    char message[45];
    char hostname[5];
    
    struct timeval time;
    struct tm* ptm;
    char time_string[40];
    
    // rank sender und empfänger
    int to;
    int from;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    MPI_Status status;
    gethostname(hostname, sizeof hostname);

    
    gettimeofday(&time, (struct timezone *) 0);
    ptm = localtime(&time.tv_sec);
    strftime(time_string, sizeof(time_string), "%Y-%m-%d %H:%M:%S", ptm);
    if (rank == 0) {
        // falls der rank 0 ist soll die microsekunden nicht mit in
        // reduce funktion betrachtet werden, da aber jeder prozess die
        // funktion ausführen muss, setzten wir hier die microseconds auf int max value
        microseconds[0] = INT_MAX;
    }
    else {
        microseconds[0] = (int) time.tv_usec; 
    }
    MPI_Reduce(microseconds, global_minimum_microseconds, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        for (from = 1; from < size; from++) {
            MPI_Recv (message, 45, MPI_CHAR, from, 0, MPI_COMM_WORLD, &status);
            printf("%s\n", message);
        }
        printf("%d\n", global_minimum_microseconds[0]);
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
