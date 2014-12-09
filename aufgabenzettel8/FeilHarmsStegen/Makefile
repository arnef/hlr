# Common definitions
CC = gcc

# Compiler flags, paths and libraries
CFLAGS = -std=c99 -pedantic -Wall -Wextra -O2 -ggdb
LFLAGS = $(CFLAGS)
LIBS   = -lm

OBJS = partdiff-mpi.o askparams.o

# Rule to create *.o from *.c
.c.o:
	mpicc -c $(CFLAGS) $*.c

# Targets ...
all: partdiff-mpi

partdiff-mpi: $(OBJS) Makefile
	mpicc $(LFLAGS) -o $@ $(OBJS) $(LIBS)

clean:
	$(RM) *.o *~

partdiff-mpi.o: partdiff-mpi.c Makefile

askparams.o: askparams.c Makefile
