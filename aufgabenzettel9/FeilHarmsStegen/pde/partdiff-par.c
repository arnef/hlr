/****************************************************************************/
/****************************************************************************/
/**                                                                        **/
/**                TU Muenchen - Institut fuer Informatik                  **/
/**                                                                        **/
/** Copyright: Prof. Dr. Thomas Ludwig                                     **/
/**            Andreas C. Schmidt                                          **/
/**                                                                        **/
/** File:      partdiff-seq.c                                              **/
/**                                                                        **/
/** Purpose:   Partial differential equation solver for Gauss-Seidel and   **/
/**            Jacobi method.                                              **/
/**                                                                        **/
/****************************************************************************/
/****************************************************************************/

/* ************************************************************************ */
/* Include standard header file.                                            */
/* ************************************************************************ */
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <malloc.h>
#include <sys/time.h>
#include <assert.h>

// --------
// Die folgenden paar Zeilen sind ausschließlich für den Syntaxchecker meines Editors bestimmt.
// Sie werden beim Compilieren ignoriert => Einfach ignorieren
#ifdef __cplusplus
#undef __cplusplus
typedef unsigned char _Bool;
#endif
// --------

#include <mpi/mpi.h>

#include "partdiff-par.h"

#define INTEGER_DIVISION_ROUNDING_UP(dividend, divisor) (((dividend)+(divisor)-1)/(divisor))

struct mpi_options
{
	uint32_t mpi_size;       /* number of mpi processes                        */
	uint32_t mpi_rank;       /* rank of this process                           */
	uint32_t num_procs_used; /* number of mpi processes actually used          */
	MPI_Comm comm;			 /* comm */
};

#define IS_LAST(mpi_options_ptr) ((mpi_options_ptr)->mpi_rank + 1 == (mpi_options_ptr)->num_procs_used)
#define IS_FIRST(mpi_options_ptr) ((mpi_options_ptr)->mpi_rank == 0)
#define NEXT_RANK(mpi_options_ptr) ((mpi_options_ptr)->mpi_rank + 1)
#define PREVIOUS_RANK(mpi_options_ptr) ((mpi_options_ptr)->mpi_rank - 1)
#define LAST_RANK(mpi_options_ptr) ((mpi_options_ptr)->num_procs_used - 1)

struct calculation_arguments
{
	uint64_t  N;              /* number of spaces between lines (lines=N+1)     */
	uint64_t  num_matrices;   /* number of matrices                             */
	double    h;              /* length of a space between two lines            */
	double    ***Matrix;      /* index matrix used for addressing M             */
	double    *M;             /* two matrices with real values                  */
	uint64_t  row_start;      /*                                                */
	uint64_t  row_end;        /*                                                */
};

struct calculation_results
{
	uint64_t  m;
	uint64_t  stat_iteration; /* number of current iteration                    */
	double    stat_precision; /* actual precision of all slaves in iteration    */
};

/* ************************************************************************ */
/* Global variables                                                         */
/* ************************************************************************ */

/* time measurement variables */
struct timeval start_time;       /* time when program started                      */
struct timeval comp_time;        /* time when calculation completed                */


/* ************************************************************************ */
/* initVariables: Initializes some global variables                         */
/* ************************************************************************ */
static
void
initVariables (struct calculation_arguments* arguments, struct calculation_results* results, struct options const* options, struct mpi_options* mpi_options)
{
	arguments->N = (options->interlines * 8) + 9 - 1;
	arguments->num_matrices = (options->method == METH_JACOBI) ? 2 : 1;
	arguments->h = 1.0 / arguments->N;

	// paralell assign rows to the processes
	uint64_t N = arguments->N;
	if(N < mpi_options->num_procs_used)
	{
		mpi_options->num_procs_used = N;
	}
	uint64_t rows_per_process = (N-1)/mpi_options->num_procs_used;
	uint64_t remaining_rows = (N-1) - mpi_options->num_procs_used * rows_per_process;
	
	// calculate how many rows each process gets
	if(mpi_options->mpi_rank < mpi_options->num_procs_used)
	{	
		uint64_t start = mpi_options->mpi_rank * rows_per_process;
		start += (mpi_options->mpi_rank < remaining_rows)? mpi_options->mpi_rank : remaining_rows;
		uint64_t end = start + rows_per_process;
		end += (mpi_options->mpi_rank < remaining_rows)? 1:0;
		assert(start <= end);
		assert(end <= N);
		arguments->row_start = start + 1;//1 is first index
		arguments->row_end = end + 1;//1 is first index
	}

	if(mpi_options->mpi_size > mpi_options->num_procs_used)
	{
		MPI_Group world_group;
		MPI_Comm_group(MPI_COMM_WORLD, &world_group);

		// create group with only necessary ranks
		int ranks[3] = {0, mpi_options->num_procs_used-1, 1};
		MPI_Group new_group;
		MPI_Group_range_incl(world_group, 1, &ranks, &new_group);

		// Create a new communicator
		MPI_Comm_create(MPI_COMM_WORLD, new_group, &mpi_options->comm);
	}

	if(mpi_options->mpi_rank >= mpi_options->num_procs_used)
	{	
		// calculate nothing
		//arguments->row_start = 0;
		//arguments->row_end = 0;
		MPI_Finalize();
		exit(0);
	}

	if(mpi_options->num_procs_used > 1)
	{
	    //
		int send_buffer_size = 1024 + 2*sizeof(double)*(N+1);
		void* send_buffer = malloc(send_buffer_size);
		MPI_Buffer_attach(send_buffer, send_buffer_size);
	}

	results->m = 0;
	results->stat_iteration = 0;
	results->stat_precision = 0;
}

/* ************************************************************************ */
/* freeMatrices: frees memory for matrices                                  */
/* ************************************************************************ */
static
void
freeMatrices (struct calculation_arguments* arguments)
{
	uint64_t i;

	for (i = 0; i < arguments->num_matrices; i++)
	{
		free(arguments->Matrix[i]);
	}

	free(arguments->Matrix);
	free(arguments->M);
}

/* ************************************************************************ */
/* allocateMemory ()                                                        */
/* allocates memory and quits if there was a memory allocation problem      */
/* ************************************************************************ */
static
void*
allocateMemory (size_t size)
{
	void *p;

	if ((p = malloc(size)) == NULL)
	{
		printf("Speicherprobleme! (%" PRIu64 " Bytes)\n", size);
		/* exit program */
		exit(1);
	}

	return p;
}

/* ************************************************************************ */
/* allocateMatrices: allocates memory for matrices                          */
/* ************************************************************************ */
static
void
allocateMatrices (struct calculation_arguments* arguments)
{
	uint64_t i, j;

	uint64_t const N = arguments->N;
	uint64_t num_rows = arguments->row_end - arguments->row_start + 2;

	//Mein Editor meckert über implizites Casten von void* nach foo* --> Explizite Casts hinzugefügt
	uint64_t buffer_length = arguments->num_matrices * num_rows * (N + 1) * sizeof(double);
	arguments->M = (double*)allocateMemory((size_t)buffer_length);
	arguments->Matrix = (double***)allocateMemory(arguments->num_matrices * sizeof(double**));

	double* p = arguments->M;
	for (i = 0; i < arguments->num_matrices; i++)
	{
		arguments->Matrix[i] = (double**)allocateMemory((N + 1) * sizeof(double*));

		for (j = 0; j <= N; j++)
		{
			if(j+1 < arguments->row_start || j > arguments->row_end)
			{
				arguments->Matrix[i][j] = NULL;
			}
			else
			{
				arguments->Matrix[i][j] = p;
				p += N+1;
			}
		}
	}
	assert(p <= arguments->M + buffer_length);
}

/* ************************************************************************ */
/* initMatrices: Initialize matrix/matrices and some global variables       */
/* ************************************************************************ */
static
void
initMatrices (struct calculation_arguments* arguments, struct options const* options)
{
	uint64_t g, i, j;                                /*  local variables for loops   */

	uint64_t const N = arguments->N;
	double const h = arguments->h;
	double*** Matrix = arguments->Matrix;

	/* initialize matrix/matrices with zeros */
	for (g = 0; g < arguments->num_matrices; g++)
	{
		for (i = 0; i <= N; i++)
		{
			if(Matrix[g][i])
			{
				for (j = 0; j <= N; j++)
				{
					Matrix[g][i][j] = 0.0;
				}
			}
		}
	}

	/* initialize borders, depending on function (function 2: nothing to do) */
	if (options->inf_func == FUNC_F0)
	{
		for (g = 0; g < arguments->num_matrices; g++)
		{
			for (i = 0; i <= N; i++)
			{
				if(Matrix[g][i]) Matrix[g][i][0] = 1.0 - (h * i);
				if(Matrix[g][i]) Matrix[g][i][N] = h * i;
				if(Matrix[g][0]) Matrix[g][0][i] = 1.0 - (h * i);
				if(Matrix[g][N]) Matrix[g][N][i] = h * i;
			}

			if(Matrix[g][N]) Matrix[g][N][0] = 0.0;
			if(Matrix[g][0]) Matrix[g][0][N] = 0.0;
		}
	}
}

static
void
wait_for_completion(MPI_Request* request)
{
	if(*request)
	{
		MPI_Status status;
		MPI_Wait(request, &status);
		*request = NULL;
	}
}

/* ************************************************************************ */
/* calculate: solves the equation using the gauss method                   */
/* ************************************************************************ */
static
void
calculate_gauss (struct calculation_arguments const* arguments, struct calculation_results *results, struct options const* options, struct mpi_options* mpi_options)
{
	const int end_tag = 41;
	uint32_t i, j;                                   /* local variables for loops  */
	int m1, m2;                                 /* used as indices for old and new matrices       */
	double star;                                /* four times center value minus 4 neigh.b values */
	double residuum;                            /* residuum of current iteration                  */
	double maxresiduum;                         /* maximum residuum value of a slave in iteration */
	

	uint32_t const N = arguments->N;
	double const h = arguments->h;

	double pih = 0.0;
	double fpisin = 0.0;

	int term_iteration = options->term_iteration;

	/* initialize m1 and m2 for gauss */
		m1 = 0;
		m2 = 0;

	if (options->inf_func == FUNC_FPISIN)
	{
		pih = PI * h;
		fpisin = 0.25 * TWO_PI_SQUARE * h * h;
	}

	MPI_Request sending_first_row_to_previous = NULL;
	MPI_Request sending_last_row_to_next = NULL;
	MPI_Request receiving_last_row_from_next = NULL;
	MPI_Request receive_precision_notification = NULL;

	uint64_t start = arguments->row_start;
	uint64_t end = arguments->row_end;
	bool is_first_iteration = true;
	bool is_precision_sufficcient = false;

  	// checking for termination by precissionflag is set in the last process for
	// absolute termination
	if(options->termination == TERM_PREC && mpi_options->num_procs_used > 1 && IS_FIRST(mpi_options))
	{
		MPI_Irecv(&is_precision_sufficcient, 1, MPI_C_BOOL, LAST_RANK(mpi_options), end_tag, mpi_options->comm, &receive_precision_notification);
	}

	while (term_iteration > 0)
	{
		double** Matrix_Out = arguments->Matrix[m1];
		double** Matrix_In  = arguments->Matrix[m2];
		bool calculate_risiuum = options->termination == TERM_PREC || term_iteration == 1;

		maxresiduum = 0;

		if(mpi_options->num_procs_used > 1)
		{
			// looking if the previous rank has set the precission sufficient flag
      			if(options->termination == TERM_PREC)
			{
				if(!IS_FIRST(mpi_options))
				{ 
          				// only can recive the flag when the privious rank has a sufficiant
          				// precision
					MPI_Recv(&is_precision_sufficcient, 1, MPI_C_BOOL, PREVIOUS_RANK(mpi_options), end_tag, mpi_options->comm, NULL);
				}
				else
				{
					MPI_Status status;
					int flag = 0;
					MPI_Test(&receive_precision_notification, &flag, &status);
					if(flag)
					{
						wait_for_completion(&receive_precision_notification);
					}
				}
			        // when the precission is sufficcient in the last iteration send the
			        // flag to the next rank; and terminate after sending that the
			        // programm can terminate
				if(is_precision_sufficcient)
				{
					if(mpi_options->mpi_rank +1 < mpi_options->num_procs_used)
						MPI_Send(&is_precision_sufficcient, 1,   MPI_C_BOOL, NEXT_RANK(mpi_options), end_tag, mpi_options->comm);
					break;
				}
			}
			if(!IS_FIRST(mpi_options))
			{
				// first own row and maxresidium (when residuum is calculated ) from previous 
				if(calculate_risiuum) MPI_Recv(&maxresiduum , 1, MPI_DOUBLE, PREVIOUS_RANK(mpi_options), 2, mpi_options->comm, NULL);
				MPI_Recv(Matrix_In[arguments->row_start-1], N+1, MPI_DOUBLE, PREVIOUS_RANK(mpi_options), 1, mpi_options->comm, NULL);
			}

			if(!is_first_iteration && !IS_LAST(mpi_options))
			{
				// receive last own row from the next process
				MPI_Irecv(Matrix_In[arguments->row_end], N+1, MPI_DOUBLE, NEXT_RANK(mpi_options), 1, mpi_options->comm, &receiving_last_row_from_next);
			}
      
      			// wait until the the following process did send the whole row
			wait_for_completion(&sending_first_row_to_previous);
		}

		/* over all rows of the process*/
		for (i = start; i < end; i++)
		{
			if(i==end-1)
			{
				// before calculating last row
				wait_for_completion(&receiving_last_row_from_next);
				wait_for_completion(&sending_last_row_to_next);
			}

			double fpisin_i = 0.0;

			if (options->inf_func == FUNC_FPISIN)
			{
				fpisin_i = fpisin * sin(pih * (double)i);
			}

			/* over all columns */
			// calcualting the part of the matrix in each process
			for (j = 1; j < N; j++)
			{
				star = 0.25 * (Matrix_In[i-1][j] + Matrix_In[i][j-1] + Matrix_In[i][j+1] + Matrix_In[i+1][j]);

				if (options->inf_func == FUNC_FPISIN)
				{
					star += fpisin_i * sin(pih * (double)j);
				}

				if (calculate_risiuum)
				{
					residuum = Matrix_In[i][j] - star;
					residuum = (residuum < 0) ? -residuum : residuum;
					maxresiduum = (residuum < maxresiduum) ? maxresiduum : residuum;
				}

				Matrix_Out[i][j] = star;
			}

			if(i==start && !IS_FIRST(mpi_options))
			{
				// send to previous process and using Isend for non blocking sendoperation to avoid deadlock
				MPI_Isend(Matrix_Out[arguments->row_start], N+1, MPI_DOUBLE, PREVIOUS_RANK(mpi_options), 1, mpi_options->comm, &sending_first_row_to_previous);

			}
		}

		results->stat_iteration++;
		results->stat_precision = maxresiduum;

		/* exchange m1 and m2 */
		i = m1;
		m1 = m2;
		m2 = i;

		/* check for stopping calculation, depending on termination method */
		
		// termination after precision
		if (options->termination == TERM_PREC)
		{
			if(IS_LAST(mpi_options))
			{
				if (maxresiduum < options->term_precision)
				{
					if(mpi_options->num_procs_used > 1)
					{
						bool v = true;
						MPI_Send(&v, 1, MPI_C_BOOL, 0, end_tag, mpi_options->comm);
					}
					else
					{
						term_iteration = 0;
					}
				}
			}
			else
			{
				bool v = false;
				MPI_Send(&v, 1, MPI_C_BOOL, NEXT_RANK(mpi_options), end_tag, mpi_options->comm);
			}
		}
		// termination with iterations
		else if (options->termination == TERM_ITER)
		{
			term_iteration--;
		}
		if(mpi_options->num_procs_used > 1)
		{
			// Rank 0 has no previous process
			// if(!IS_FIRST(mpi_options))
			// {
			// 	// send to previous process and using Isend for non blocking sendoperation to avoid deadlock
			// 	MPI_Isend(Matrix_Out[arguments->row_start], N+1, MPI_DOUBLE, PREVIOUS_RANK(mpi_options), 1, mpi_options->comm, &sending_first_row_to_previous);
			// }
			// the last process has no following process
			if(!IS_LAST(mpi_options))
			{
				if(calculate_risiuum) MPI_Bsend(&maxresiduum, 1, MPI_DOUBLE, NEXT_RANK(mpi_options), 2, mpi_options->comm);
				// send to next process and using Isend for non blocking sendoperation to avoid deadlock
				MPI_Isend(Matrix_Out[arguments->row_end-1], N+1, MPI_DOUBLE, NEXT_RANK(mpi_options), 1, mpi_options->comm, &sending_last_row_to_next);
			}
		}
		is_first_iteration = false;
	}//while

	if(mpi_options->num_procs_used > 1)
	{
		if(!is_first_iteration && !IS_LAST(mpi_options))
		{
			MPI_Recv(arguments->Matrix[m2][end], N+1, MPI_DOUBLE, NEXT_RANK(mpi_options), 1, mpi_options->comm, NULL);
		}
		wait_for_completion(&sending_first_row_to_previous);
		wait_for_completion(&sending_last_row_to_next);
		wait_for_completion(&receiving_last_row_from_next);
		wait_for_completion(&receive_precision_notification);

		//rank 0 displays results, not LAST_RANK
		if(IS_FIRST(mpi_options))
		{
			MPI_Recv(&results->stat_precision, 1, MPI_DOUBLE, LAST_RANK(mpi_options), 0, mpi_options->comm, NULL);
		}
		else if(IS_LAST(mpi_options))
		{
			MPI_Send(&results->stat_precision, 1, MPI_DOUBLE, 0, 0, mpi_options->comm);
		}
	}

	results->m = m2;
}

static
void
calculate_jacobi (struct calculation_arguments const* arguments, struct calculation_results *results, struct options const* options, struct mpi_options* mpi_options)
{
	uint32_t i, j;                                   /* local variables for loops  */
	int m1, m2;                                 /* used as indices for old and new matrices       */
	double star;                                /* four times center value minus 4 neigh.b values */
	double residuum;                            /* residuum of current iteration                  */
	double maxresiduum;                         /* maximum residuum value of a slave in iteration */
	

	uint32_t const N = arguments->N;
	double const h = arguments->h;

	double pih = 0.0;
	double fpisin = 0.0;

	int term_iteration = options->term_iteration;

	/* initialize m1 and m2 for jacobi */
		m1 = 0;
		m2 = 1;

	if (options->inf_func == FUNC_FPISIN)
	{
		pih = PI * h;
		fpisin = 0.25 * TWO_PI_SQUARE * h * h;
	}

	uint64_t start = arguments->row_start;
	uint64_t end = arguments->row_end;

	while (term_iteration > 0)
	{
		double** Matrix_Out = arguments->Matrix[m1];
		double** Matrix_In  = arguments->Matrix[m2];
		bool calculate_risiuum = options->termination == TERM_PREC || term_iteration == 1;

		maxresiduum = 0;

		/* over all rows of the process*/
		for (i = start; i < end; i++)
		{
			double fpisin_i = 0.0;

			if (options->inf_func == FUNC_FPISIN)
			{
				fpisin_i = fpisin * sin(pih * (double)i);
			}

			/* over all columns */
			for (j = 1; j < N; j++)
			{
				star = 0.25 * (Matrix_In[i-1][j] + Matrix_In[i][j-1] + Matrix_In[i][j+1] + Matrix_In[i+1][j]);

				if (options->inf_func == FUNC_FPISIN)
				{
					star += fpisin_i * sin(pih * (double)j);
				}

				if (calculate_risiuum)
				{
					residuum = Matrix_In[i][j] - star;
					residuum = (residuum < 0) ? -residuum : residuum;
					maxresiduum = (residuum < maxresiduum) ? maxresiduum : residuum;
				}

				Matrix_Out[i][j] = star;
			}
		}

		if(calculate_risiuum && mpi_options->num_procs_used > 1)
		{
			//assign global residuum to maxresiduum
			double local_risiuum = maxresiduum;
			MPI_Allreduce(&local_risiuum, &maxresiduum, 1, MPI_DOUBLE, MPI_MAX, mpi_options->comm);
			
		}
		

		results->stat_iteration++;
		results->stat_precision = maxresiduum;

		/* exchange m1 and m2 */
		i = m1;
		m1 = m2;
		m2 = i;

		/* check for stopping calculation, depending on termination method */
		if (options->termination == TERM_PREC)
		{
			if (maxresiduum < options->term_precision)
			{
				term_iteration = 0;
			}
		}
		else if (options->termination == TERM_ITER)
		{
			term_iteration--;
		}
		// Sending the rows needed by other processes
		if(term_iteration > 0)
		{
			MPI_Request request1 = NULL;
			MPI_Request request2 = NULL;
			
			// Rank 0 has no previous process
			if(mpi_options->mpi_rank > 0)
			{
				// send to previous process and using Isend for non blocking sendoperation to avoid deadlock
				MPI_Isend(Matrix_Out[arguments->row_start], N, MPI_DOUBLE, mpi_options->mpi_rank-1, 1, mpi_options->comm, &request1);
			}
			// the last process has no following process
			if(mpi_options->mpi_rank+1 < mpi_options->num_procs_used)
			{
				// send to next process and using Isend for non blocking sendoperation to avoid deadlock
				MPI_Isend(Matrix_Out[arguments->row_end-1], N, MPI_DOUBLE, mpi_options->mpi_rank+1, 1, mpi_options->comm, &request2);
			}

			// Rank 0 has no previous process
			if(mpi_options->mpi_rank > 0)
			{
				// receive from previous process
				MPI_Recv(Matrix_Out[arguments->row_start-1], N, MPI_DOUBLE, mpi_options->mpi_rank-1, 1, mpi_options->comm, NULL);
			}
			// the last process has no following process
			if(mpi_options->mpi_rank+1 < mpi_options->num_procs_used)
			{
				// receive from next process
				MPI_Recv(Matrix_Out[arguments->row_end], N, MPI_DOUBLE, mpi_options->mpi_rank+1, 1, mpi_options->comm, NULL);
			}
			MPI_Status status;
			if(request1) MPI_Wait(&request1, &status);
			if(request2) MPI_Wait(&request2, &status);
		}
	}

	results->m = m2;
}

/* ************************************************************************ */
/*  displayStatistics: displays some statistics about the calculation       */
/* ************************************************************************ */
static
void
displayStatistics (struct calculation_arguments const* arguments, struct calculation_results const* results, struct options const* options)
{
	int N = arguments->N;
	double time = (comp_time.tv_sec - start_time.tv_sec) + (comp_time.tv_usec - start_time.tv_usec) * 1e-6;

	printf("Berechnungszeit:    %f s \n", time);
	printf("Speicherbedarf:     %f MiB\n", (N + 1) * (N + 1) * sizeof(double) * arguments->num_matrices / 1024.0 / 1024.0);
	printf("Berechnungsmethode: ");

	if (options->method == METH_GAUSS_SEIDEL)
	{
		printf("Gauss-Seidel");
	}
	else if (options->method == METH_JACOBI)
	{
		printf("Jacobi");
	}

	printf("\n");
	printf("Interlines:         %" PRIu64 "\n",options->interlines);
	printf("Stoerfunktion:      ");

	if (options->inf_func == FUNC_F0)
	{
		printf("f(x,y) = 0");
	}
	else if (options->inf_func == FUNC_FPISIN)
	{
		printf("f(x,y) = 2pi^2*sin(pi*x)sin(pi*y)");
	}

	printf("\n");
	printf("Terminierung:       ");

	if (options->termination == TERM_PREC)
	{
		printf("Hinreichende Genaugkeit");
	}
	else if (options->termination == TERM_ITER)
	{
		printf("Anzahl der Iterationen");
	}

	printf("\n");
	printf("Anzahl Iterationen: %" PRIu64 "\n", results->stat_iteration);
	printf("Norm des Fehlers:   %e\n", results->stat_precision);
	printf("\n");
}

/**
 * rank and size are the MPI rank and size, respectively.
 * from and to denote the global(!) range of lines that this process is responsible for.
 *
 * Example with 9 matrix lines and 4 processes:
 * - rank 0 is responsible for 1-2, rank 1 for 3-4, rank 2 for 5-6 and rank 3 for 7.
 *   Lines 0 and 8 are not included because they are not calculated.
 * - Each process stores two halo lines in its matrix (except for ranks 0 and 3 that only store one).
 * - For instance: Rank 2 has four lines 0-3 but only calculates 1-2 because 0 and 3 are halo lines for other processes. It is responsible for (global) lines 5-6.
 */
static
void
DisplayMatrix (struct calculation_arguments* arguments, struct calculation_results* results, struct options* options, int rank, int size, int from, int to)
{
  int const elements = 8 * options->interlines + 9;

  int x, y;
  double** Matrix = arguments->Matrix[results->m];
  MPI_Status status;

  /* first line belongs to rank 0 */
  if (rank == 0)
    from--;

  /* last line belongs to rank size - 1 */
  if (rank + 1 == size)
    to++;

  if (rank == 0)
    printf("Matrix:\n");

  for (y = 0; y < 9; y++)
  {
    int line = y * (options->interlines + 1);

    if (rank == 0)
    {
      /* check whether this line belongs to rank 0 */
      if (line < from || line > to)
      {
        /* use the tag to receive the lines in the correct order
         * the line is stored in Matrix[0], because we do not need it anymore */
        MPI_Recv(Matrix[0], elements, MPI_DOUBLE, MPI_ANY_SOURCE, 42 + y, MPI_COMM_WORLD, &status);
      }
    }
    else
    {
      if (line >= from && line <= to)
      {
        /* if the line belongs to this process, send it to rank 0
         * (line - from + 1) is used to calculate the correct local address */
        MPI_Send(Matrix[line], elements, MPI_DOUBLE, 0, 42 + y, MPI_COMM_WORLD);
      }
    }

    if (rank == 0)
    {
      for (x = 0; x < 9; x++)
      {
        int col = x * (options->interlines + 1);

        if (line >= from && line <= to)
        {
          /* this line belongs to rank 0 */
          printf("%7.4f", Matrix[line][col]);
        }
        else
        {
          /* this line belongs to another rank and was received above */
          printf("%7.4f", Matrix[0][col]);
        }
      }

      printf("\n");
    }
  }

  fflush(stdout);
}


/****************************************************************************/
/** Beschreibung der Funktion DisplayMatrix:                               **/
/**                                                                        **/
/** Die Funktion DisplayMatrix gibt eine Matrix                            **/
/** in einer "ubersichtlichen Art und Weise auf die Standardausgabe aus.   **/
/**                                                                        **/
/** Die "Ubersichtlichkeit wird erreicht, indem nur ein Teil der Matrix    **/
/** ausgegeben wird. Aus der Matrix werden die Randzeilen/-spalten sowie   **/
/** sieben Zwischenzeilen ausgegeben.                                      **/
/****************************************************************************/
// static
// void
// DisplayMatrix (struct calculation_arguments* arguments, struct calculation_results* results, struct options* options)
// {
// 	int x, y;

// 	double** Matrix = arguments->Matrix[results->m];

// 	int const interlines = options->interlines;

// 	printf("Matrix:\n");

// 	for (y = 0; y < 9; y++)
// 	{
// 		for (x = 0; x < 9; x++)
// 		{
// 			printf ("%7.4f", Matrix[y * (interlines + 1)][x * (interlines + 1)]);
// 		}

// 		printf ("\n");
// 	}

// 	fflush (stdout);
// }



/* ************************************************************************ */
/*                                                                          */
/* ************************************************************************ */
void
initMpi(struct mpi_options* mpi_options, int* argc, char*** argv)
{
	/* default values if mpi is not used */
	mpi_options->mpi_rank = 0;
	mpi_options->mpi_size = 1;

	/* initialize mpi */
	MPI_Init(argc, argv);
	MPI_Comm_rank (MPI_COMM_WORLD, (int*)&mpi_options->mpi_rank);
	MPI_Comm_size (MPI_COMM_WORLD, (int*)&mpi_options->mpi_size);
	mpi_options->comm = MPI_COMM_WORLD;
	mpi_options->num_procs_used = mpi_options->mpi_size;
}

/* ************************************************************************ */
/*  main                                                                    */
/* ************************************************************************ */
int
main (int argc, char** argv)
{
	struct options options;
	struct mpi_options mpi_options;
	struct calculation_arguments arguments;
	struct calculation_results results;
	
	initMpi(&mpi_options, &argc, &argv);
	/* get parameters */
	AskParams(&options, argc, argv, mpi_options.mpi_rank == 0);              /* ************************* */

	initVariables(&arguments, &results, &options, &mpi_options);           /* ******************************************* */
	allocateMatrices(&arguments);        /*  get and initialize variables and matrices  */
	initMatrices(&arguments, &options);            /* ******************************************* */

	gettimeofday(&start_time, NULL);                   /*  start timer         */
	if(options.method == METH_GAUSS_SEIDEL)
	{
		calculate_gauss(&arguments, &results, &options, &mpi_options);
	}
	else
	{
		calculate_jacobi(&arguments, &results, &options, &mpi_options);
	}
	gettimeofday(&comp_time, NULL);                   /*  stop timer          */

	if(mpi_options.mpi_rank == 0) displayStatistics(&arguments, &results, &options);
	DisplayMatrix(&arguments, &results, &options, mpi_options.mpi_rank, mpi_options.mpi_size, arguments.row_start, arguments.row_end);

	freeMatrices(&arguments);                                       /*  free memory     */

	MPI_Finalize();
	return 0;
}
