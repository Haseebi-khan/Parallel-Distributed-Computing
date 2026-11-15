#include <iostream>
#include <stdio.h>

using namespace std;

// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs$ g++ -g -Wall -O2 -fopenmp -o ./bin/macros ./src/practriceofmacros.cpp
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs$ ./bin/macros
// Program started.
// Debug mode is active.
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs$ g++ -g -Wall -O2 -fopenmp -o ./bin/macros ./src/practriceofmacros.cpp
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs$ ./bin/macros
// Program started.
// Release mode is active.
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs$ 

// #define DEBUG  

#ifdef _OPENMP
    #include <omp.h>
#endif

int main() 
{
    printf("Program started.\n");
#ifdef DEBUG
    printf("Debug mode is active.\n");
#else
    printf("Release mode is active.\n");
#endif

    return 0;
}
