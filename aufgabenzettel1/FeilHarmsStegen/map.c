#include <stdio.h>

// Definieren Sie ein enum cardd
typedef enum { N = 1, E=2, S=4, W=8 } cardd;

// Definieren Sie ein 3x3-Array namens map, das Werte vom Typ cardd enthält
static cardd map[3][3] = {
	{-1, -1, -1 },
	{-1, -1, -1 },
	{-1, -1, -1}
};

// Die Funktion set_dir soll an Position x, y den Wert dir in das Array map eintragen
// Überprüfen Sie x und y um mögliche Arrayüberläufe zu verhindern
// Überprüfen Sie außerdem dir auf Gültigkeit
void set_dir (int x, int y, cardd dir)
{
	//TODO gültigkeit von dir
	if (x < 3 && y < 3) {
		map[x][y] = dir;
	}
}

// Die Funktion show_map soll das Array in Form einer 3x3-Matrix ausgeben
void show_map (void)
{
	int i,j;
	// zeilen
	for (i = 0; i < 3; ++i) {
		// spalten
		for (j = 0; j < 3; ++j) {
			switch (map[i][j]) {
				case N: 
					printf("N");
					break;
				case E: 
					printf("E");
					break;
				case W: 
					printf("W");
					break;
				case S: 
					printf("S");
					break;
				case 9:
					printf("NW");
					break;
				case 3:
					printf("NE");
					break;
				case 5:
					printf("NE");
					break;
				case 12:
					printf("SW");
					break;
				case 6:
					printf("SE");
					break;
				case 10:
					printf("EW");
					break;
				default: 
					printf("0");
					break;
			}
			if (i % 2 == 0) {
				printf("  ");
			}
			else {
				printf("   ");
			}
		}
		printf("\n");
	}
}

int main (void)
{
	// In dieser Funktion darf nichts verändert werden!
	set_dir(0, 1, N);
	set_dir(1, 0, W);
	set_dir(1, 4, W);
	set_dir(1, 2, E);
	set_dir(2, 1, S);

	set_dir(0, 0, N|W);
	set_dir(0, 2, N|E);
	set_dir(0, 2, N|S);
	set_dir(2, 0, S|W);
	set_dir(2, 2, S|E);
	set_dir(2, 2, E|W);

	show_map();

	return 0;
}
