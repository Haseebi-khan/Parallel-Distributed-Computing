#include <iostream>
#include <omp.h>

using namespace std;

int main() 
{

    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        cout << "Hello: Thread " << id << endl;
        cout << "World: Thread " << id << endl;
    }

    return 0;
}



// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc$ g++ -g -Wall -fopenmp -o ex1 ex1.cpp
// ./ex1
// Hello: Thread Hello: Thread Hello: Thread 1196
// World: Thread 9
// Hello: Thread Hello: Thread 410
// World: Thread 10
// Hello: Thread Hello: Thread 

// Hello: Thread World: Thread 4
// Hello: Thread 7
// World: Thread 7
// Hello: Thread 1
// World: Thread 1
// World: Thread 118
// 2
// World: Thread 2

// World: Thread 6

// World: Thread 8
// Hello: Thread 0
// World: Thread 0
// 3
// World: Thread 3
// Hello: Thread 5
// World: Thread 5
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc$ g++ -g -Wall -fopenmp -o ex1 ex1.cpp