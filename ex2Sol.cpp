#include <iostream>
#include <omp.h>
using namespace std;

static long num_step = 900000000;

double step;

int main(int argc, char *argv[]){


    double pi, sum = 0.0;
    step = 1.0 / (double)num_step;

    int thread_count = 6; 

    double start = omp_get_wtime();
    #pragma opm parallel for num_threads(thread_count) reduction(+:sum)
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

// OUTPUT:
// int thread_count = 6;  
// n = 100000 
// Time Taken: 0.00172641
// Calculated Value: 3.14159




// OUTPUT:
// int thread_count = 6; 
// n = 900000000 
// Time Taken: 10.6992
// Calculated Value: 3.14159