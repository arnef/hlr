
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <assert.h>
#include <stdbool.h>
#undef __cplusplus
#include <mpi/mpi.h>

#define INTEGER_DIVISION_ROUNDING_UP(dividend, divisor) (((dividend)+(divisor)-1)/(divisor))

int*
init (int N)
{
  int* buf = (int*)malloc(sizeof(int) * N);

  for (int i = 0; i < N; i++)
  {
    buf[i] = rand() % 25; //do not modify %25
  }

  return buf;
}

void send_buf(int* buf, int length, int destination)
{
  MPI_Bsend(&length, 1, MPI_INT, destination, 0, MPI_COMM_WORLD);
  MPI_Bsend(buf, length, MPI_INT, destination, 0, MPI_COMM_WORLD);
}

int receive_buf(int* buf, int source)
{
  int length = -1;
  MPI_Recv(&length, 1, MPI_INT, source, 0, MPI_COMM_WORLD, NULL);
  assert(length >= 0);
  MPI_Recv(buf, length, MPI_INT, source, 0, MPI_COMM_WORLD, NULL);
  return length;
}

int
circle (int* buf, int length, int rank, int num_ranks)
{
  if(rank >= num_ranks || num_ranks == 1)
    return length; // Entweder nichts zu verschieben oder sender ist gleich empfänger
  int next_rank = (rank + 1)%num_ranks;
  int previous_rank = (rank +num_ranks -1)%num_ranks;
  
  send_buf(buf, length, next_rank);
  return receive_buf(buf, previous_rank);
}

void print_buf(int* buf, int length, int rank)
{  
    for (int i = 0; i < length; i++)
    {
      printf ("rank %d: %d\n", rank, buf[i]);
    }
}

void print_array(const char* caption, int* buf, int length, int stride, int rank, int num_ranks)
{
  if(rank >= num_ranks)
    return;
  if(rank == 0)
  {
    printf("\n%s\n", caption);
    print_buf(buf, length, 0);
    int* buf2 = (int*)malloc(sizeof(int) * stride);
    for(int i=1;i<num_ranks;i++)
    {
      int l = receive_buf(buf2, i);
      print_buf(buf2, l, i);
    }
    free(buf2);
  }
  else
  {
    send_buf(buf, length, 0);
  }
}

int
main (int argc, char** argv)
{
  int N;
  int mpi_rank = 0, mpi_size = 1;

  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size (MPI_COMM_WORLD, &mpi_size);

  srand(time(NULL) - mpi_rank);

  if ((argc-1) != 1)//argv[0] ist das Programm selbst
  {
    printf("Expected exactly one argument, but got %d\n",(argc-1));
    return EXIT_FAILURE;
  }

  //sscanf ist bekannt für buffer overflows. => Entfernt!
  const char* arg = argv[1];
  char* pEnd = NULL;

  //Gesamtlänge des arrays bestimmen:
  N = strtol(arg, &pEnd, 10);
  if(!pEnd || *pEnd)
  {
    printf("Argument is not a valid integer.\n");
    return EXIT_FAILURE;
  }
  if(N <= 0 || N > INT_MAX)
  {
    printf("Argument is too small: %d\n",N);
    return EXIT_FAILURE;
  }

  //Sendepuffer ausreichend vergrößern
  int send_buffer_size = 1024 + 4*N;
  void* send_buffer = malloc(send_buffer_size);
  MPI_Buffer_attach(send_buffer, send_buffer_size);

  //Ausrechnen, wie die Elemente verteilt werden:
  int stride = INTEGER_DIVISION_ROUNDING_UP(N, mpi_size); //Maximale Anzahl an Elementen pro Prozess
  int num_procs_needed = INTEGER_DIVISION_ROUNDING_UP(N,stride); //Maximale an Prozessen (überschüssige werden ignoriert)
  int last_proc_rank = num_procs_needed-1;

  //Ausrechnen, wie viele Elemente der Aktuelle Prozess am Anfang hat
  int start = stride * mpi_rank;
  int end = start + stride;
  if(end > N)
  {
    end = N;
  }
  //Achtung: Gegen Ende des Arrays kann length kleiner sein als stride.
  int length = end - start;
  if(length < 0)
    length = 0;

  int* buf = init(stride);

  print_array("BEFORE", buf,length,stride, mpi_rank, num_procs_needed);

  char stop_char = '\0';
  //Sende die Endmarkierung vom ersten zum letzten Prozess
  if(mpi_rank == 0)
  {
    MPI_Bsend(buf, 1, MPI_CHAR, last_proc_rank, 0, MPI_COMM_WORLD);
  }
  if(mpi_rank == last_proc_rank)
  {
    MPI_Recv(&stop_char, 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD, NULL);
  }

  int iterations;
  for(iterations = 0; true; iterations++)
  {
    //Festestellen, ob der Endzustand erreicht ist und diese Information auf alle Knoten verteilen
    bool is_done = (mpi_rank == last_proc_rank) && (buf[0] == stop_char);//bool in c :(
    MPI_Bcast(&is_done, 1, MPI_C_BOOL, last_proc_rank, MPI_COMM_WORLD);
    if(is_done)
    {
      break;
    }

    //Die Werte im Kreis verschieben:
    length = circle(buf, length, mpi_rank, num_procs_needed);
  }
  if(mpi_rank == 0)
    printf("\nFinished after %d iterations.\n",iterations);
  print_array("AFTER", buf,length,stride, mpi_rank, num_procs_needed);
  MPI_Finalize();

  return EXIT_SUCCESS;
}
