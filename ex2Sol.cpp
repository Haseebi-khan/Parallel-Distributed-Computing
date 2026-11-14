#include <iostream>
#include <omp.h>
using namespace std;

static long num_step = 100000;

double step;

int main(int argc, char *argv[]){


    double pi, sum = 0.0;
    step = 1.0 / (double)num_step;

    int thread_count = 6; 

    double start = omp_get_wtime();

    #pragma omp parallel for num_threads(thread_count) reduction(+:sum)    
    for (int i = 0; i < num_step; i++)
    {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x*x);        
    }
    double finish = omp_get_wtime();
    
    pi = step * sum;
    cout << "Time Taken: "<< finish - start <<endl; 
    cout << "Calculated Value: " << pi <<endl;

    return 0;
}

// Serial OUTPUT:
// int thread_count = 1;  
// n = 100000 
// Time Taken: 0.00102496
// Calculated Value: 3.14159

// Parallel OUTPUT:
// int thread_count = 6;  
// n = 100000 
// Time Taken: 0.000956388
// Calculated Value: 3.14159

// ---------------------------------------

// Serial OUTPUT:
// int thread_count = 1; 
// n = 900000000 
// Time Taken: 10.8486
// Calculated Value: 3.14159

// Parallel OUTPUT:
// int thread_count = 6; 
// n = 900000000 
// Time Taken: 1.88533
// Calculated Value: 3.14159