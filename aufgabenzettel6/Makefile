SRC = timempi.c timempi2.c

all: timempi timempi2

timempi: timempi.o
	mpicc -o timempi timempi.o 

timempi2: timempi2.o
	mpicc -o timempi2 timempi2.o

%.o: %.c
	mpicc -g -c $(SRC)


clean:
	rm -f timempi timempi
	rm -f *.o
	rm -f *~
