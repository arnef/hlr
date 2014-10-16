SRC = map.c pointer.c

all: pointer map

pointer: pointer.o
	gcc -o pointer pointer.o 

map: map.o
	gcc -o map map.o

%.o: %.c
	gcc -c $(SRC)


clean:
	rm -f pointer map
	rm -f *.o
	rm -f *~
