
#include <iostream>
#include <omp.h>

using namespace std;

int main()
{
    const int n = 1000000;
    const int P = 8;

    double x[n];
    double sum = 0.0;

    omp_set_dynamic(0);

    double start = omp_get_wtime();

    #pragma omp parallel num_threads(P) reduction(+:sum)
    {
        int my_rank = omp_get_thread_num();

        // Thread 0 initializes the array
        if (my_rank == 0)
        {   
        // #pragma omp single // it will cause the deadlock'
            for (int i = 0; i < n; i++)
            {
                x[i] = 1.0;
                if (i == n-1)
                {
                    cout << "My Rank: " << my_rank << endl;
                }
            }
        }

        // Synchronize all threads before summation
        #pragma omp barrier

        // All threads calculate their partial sums
        #pragma omp for
        for (int i = 0; i < n; i++)
        {
            sum += x[i];
        }
    }

    double finish = omp_get_wtime() - start;

    cout << "Total Sum: " << sum << endl;
    cout << "Execution Time: " << finish << " seconds" << endl;

    return 0;
}

// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc$ ./root
// Total Sum: 1000
// Execution Time: 0.000239365 seconds
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc$ g++ -O2 -Wall -fopenmp -o root barrire.cpp
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc$ ./root
// Total Sum: 1000
// Execution Time: 0.000245921 seconds

// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc$ ./root
// My Rank: 0
// Total Sum: 1e+06
// Execution Time: 0.00197629 seconds
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc$ ./root
// My Rank: 0
// Total Sum: 1e+06
// Execution Time: 0.0019864 seconds
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc$ ./root
// My Rank: 0
// Total Sum: 1e+06
// Execution Time: 0.00185439 seconds
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc$  