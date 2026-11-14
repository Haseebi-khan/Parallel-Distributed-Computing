#include <iostream>
#include <omp.h>
using namespace std;

// create the parallel version of the pi program using a parallel construct
// int omp_get_num_threads();
// Number of threads in the team
// int omp_get_thread_num();
// Thread ID or rank
// double omp_get_wtime();
// Time in seconds since a fixed
// point in the past

static long num_step = 900000000;

double step;

int main(int argc, char *argv[]){
    
    double x, pi, sum = 0.0;
    // casting  = (double)num_step;
    
    step = 1.0 / (double)num_step;
    double start = omp_get_wtime();
    for (int i = 0; i < num_step; i++)
    {
        x = (i + 0.5) * step;
        sum = sum + 4.0 / (1.0 + x*x);

    }
    double finish = omp_get_wtime();
    pi = step * sum;
    cout << "Time Taken: "<< finish - start <<endl; 
    cout << "Calculated Value: " << pi <<endl;


    return 0;
}

// OUTPUT:
// int thread_count = 1;  
// n = 100000 
// Time Taken: 0.00102496
// Calculated Value: 3.14159




// OUTPUT:
// int thread_count = 1; 
// n = 900000000 
// Time Taken: 10.8486
// Calculated Value: 3.14159