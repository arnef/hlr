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
#include <pthread.h>

#include "partdiff-posix.h"

struct calculation_arguments
{
	uint64_t  N;              /* number of spaces between lines (lines=N+1)     */
	uint64_t  num_matrices;   /* number of matrices                             */
	double    h;              /* length of a space between two lines            */
	double    ***Matrix;      /* index matrix used for addressing M             */
	double    *M;             /* two matrices with real values                  */
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
initVariables (struct calculation_arguments* arguments, struct calculation_results* results, struct options const* options)
{
	arguments->N = (options->interlines * 8) + 9 - 1;
	arguments->num_matrices = (options->method == METH_JACOBI) ? 2 : 1;
	arguments->h = 1.0 / arguments->N;

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

	arguments->M = allocateMemory(arguments->num_matrices * (N + 1) * (N + 1) * sizeof(double));
	arguments->Matrix = allocateMemory(arguments->num_matrices * sizeof(double**));

	for (i = 0; i < arguments->num_matrices; i++)
	{
		arguments->Matrix[i] = allocateMemory((N + 1) * sizeof(double*));

		for (j = 0; j <= N; j++)
		{
			arguments->Matrix[i][j] = arguments->M + (i * (N + 1) * (N + 1)) + (j * (N + 1));
		}
	}
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
			for (j = 0; j <= N; j++)
			{
				Matrix[g][i][j] = 0.0;
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
				Matrix[g][i][0] = 1.0 - (h * i);
				Matrix[g][i][N] = h * i;
				Matrix[g][0][i] = 1.0 - (h * i);
				Matrix[g][N][i] = h * i;
			}

			Matrix[g][N][0] = 0.0;
			Matrix[g][0][N] = 0.0;
		}
	}
}


typedef struct calculation_process
{
	pthread_mutex_t mutex;
	struct calculation_arguments const* arguments;
	struct options const* options;
	int m1, m2;                               /* used as indices for old and new matrices       */
	double pih;
	double fpisin;
	int term_iteration;
} calculation_process;

typedef struct thread_args_
{
	pthread_t pthread;
	calculation_process* calculation;
	int rowStart;
	int rowStop;
	double maxresiduum; 
} thread_args;

void* kernel(void* param)
{
	thread_args* thread = (thread_args*)param;
	calculation_process* calculation = thread->calculation;
	struct calculation_arguments const* arguments = calculation->arguments;
	struct options const* options = calculation->options;

	int const N = arguments->N;

	double** Matrix_Out = arguments->Matrix[calculation->m1];
	double** Matrix_In  = arguments->Matrix[calculation->m2];
	double maxresiduum = 0;                         /* maximum residuum value of a slave in iteration */

	/* over all rows */
	for (int i = thread->rowStart; i < thread->rowStop; i++)//for (int i = 1; i < N; i++)
	{
		double fpisin_i = 0.0;

		if (options->inf_func == FUNC_FPISIN)
		{
			fpisin_i = calculation->fpisin * sin(calculation->pih * (double)i);
		}

		/* over all columns */
		for (int j = 1; j < N; j++)
		{
			double star;                                /* four times center value minus 4 neigh.b values */
			star = 0.25 * (Matrix_In[i-1][j] + Matrix_In[i][j-1] + Matrix_In[i][j+1] + Matrix_In[i+1][j]);

			if (options->inf_func == FUNC_FPISIN)
			{
				star += fpisin_i * sin(calculation->pih * (double)j);
			}

			if (options->termination == TERM_PREC || calculation->term_iteration == 1)
			{
				double residuum = Matrix_In[i][j] - star; /* residuum of current iteration                  */
				residuum = (residuum < 0) ? -residuum : residuum;
				maxresiduum = (residuum < maxresiduum) ? maxresiduum : residuum;
			}

			Matrix_Out[i][j] = star;
		}
	}
	thread->maxresiduum = maxresiduum;
}

/* ************************************************************************ */
/* calculate: solves the equation                                           */
/* ************************************************************************ */
static
void
calculate (struct calculation_arguments const* arguments, struct calculation_results *results, struct options const* options)
{
	calculation_process calculation;
	calculation.arguments = arguments;
	calculation.options = options;
	calculation.term_iteration = options->term_iteration;
	calculation.fpisin = 0.0;
	calculation.pih = 0;

	int const N = arguments->N;
	double const h = arguments->h;

	/* initialize m1 and m2 depending on algorithm */
	if (options->method == METH_JACOBI)
	{
		calculation.m1 = 0;
		calculation.m2 = 1;
	}
	else
	{
		calculation.m1 = 0;
		calculation.m2 = 0;
	}

	if (options->inf_func == FUNC_FPISIN)
	{
		calculation.pih = PI * h;
		calculation.fpisin = 0.25 * TWO_PI_SQUARE * h * h;
	}


	//Initialisiere Threads
	int num_threads = options->number;
	int rows_per_thread = (N + num_threads-1)/num_threads;
	int row = 1;

	thread_args* threads = (thread_args*)calloc(num_threads, sizeof(thread_args));
	for(int i = 0; i < num_threads; i++)
	{
		thread_args* thread = threads+i;
		thread->calculation = &calculation;
		thread->rowStart = row;
		row += rows_per_thread;
		if(row > N)
			row = N;
		thread->rowStop = row;
	}

	while (calculation.term_iteration > 0)
	{
		for(int i=0; i < num_threads; i++)
		{
			pthread_create(&threads[i].pthread, NULL, &kernel, &threads[i]);
		}

		double maxresiduum = 0;
		for(int i=0; i< num_threads; i++)
		{
			void* result;
			pthread_join(threads[i].pthread, &result);
			//pthread_detach(&threads[i].pthread)

			if(maxresiduum < threads[i].maxresiduum)
				maxresiduum = threads[i].maxresiduum;
		}

		results->stat_iteration++;
		results->stat_precision = maxresiduum;

		/* exchange m1 and m2 */
		int mh = calculation.m1;
		calculation.m1 = calculation.m2;
		calculation.m2 = mh;
		/* check for stopping calculation, depending on termination method */
		if (options->termination == TERM_PREC)
		{
			if (maxresiduum < options->term_precision)
			{
				calculation.term_iteration = 0;
			}
		}
		else if (options->termination == TERM_ITER)
		{
			calculation.term_iteration--;
		}
	}
	free(threads);

	results->m = calculation.m2;
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
static
void
DisplayMatrix (struct calculation_arguments* arguments, struct calculation_results* results, struct options* options)
{
	int x, y;

	double** Matrix = arguments->Matrix[results->m];

	int const interlines = options->interlines;

	printf("Matrix:\n");

	for (y = 0; y < 9; y++)
	{
		for (x = 0; x < 9; x++)
		{
			printf ("%7.4f", Matrix[y * (interlines + 1)][x * (interlines + 1)]);
		}

		printf ("\n");
	}

	fflush (stdout);
}

/* ************************************************************************ */
/*  main                                                                    */
/* ************************************************************************ */
int
main (int argc, char** argv)
{
	struct options options;
	struct calculation_arguments arguments;
	struct calculation_results results;

	/* get parameters */
	AskParams(&options, argc, argv);              /* ************************* */

	initVariables(&arguments, &results, &options);           /* ******************************************* */

	allocateMatrices(&arguments);        /*  get and initialize variables and matrices  */
	initMatrices(&arguments, &options);            /* ******************************************* */

	gettimeofday(&start_time, NULL);                   /*  start timer         */
	calculate(&arguments, &results, &options);                                      /*  solve the equation  */
	gettimeofday(&comp_time, NULL);                   /*  stop timer          */

	displayStatistics(&arguments, &results, &options);
	DisplayMatrix(&arguments, &results, &options);

	freeMatrices(&arguments);                                       /*  free memory     */

	return 0;
}
