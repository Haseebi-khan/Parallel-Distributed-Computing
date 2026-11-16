
#include <iostream>
#include <omp.h>
#include <vector>

using namespace std;

int main(int argc, char *argv[])
{
    int N = 500000000;
    int P = 8;

    vector<int> a(N);

    for (int i = 0; i < N; i++)
    {
        a[i] = i + 1;
    }

    omp_set_dynamic(0);

    // ==================================
    // My Solution: Parallel For + Reduction
    // ==================================

    double sum1 = 0;

    double start = omp_get_wtime();

    #pragma omp parallel for num_threads(P) reduction(+:sum1)
    for (int i = 0; i < N; i++)
    {
        sum1 += a[i];
    }

    double finish = omp_get_wtime() - start;

    cout << "\n========== My Solution ==========" << endl;
    cout << "Total Sum: " << sum1 << endl;
    cout << "Execution Time: " << finish << " seconds" << endl;


    // ==================================
    // Book Solution: Balanced Block Partition
    // ==================================

    double sum2 = 0;

    start = omp_get_wtime();

    #pragma omp parallel num_threads(P) reduction(+:sum2)
    {
        int my_rank = omp_get_thread_num();
        int thread_count = omp_get_num_threads();

        // Balanced block partition
        int q = N / thread_count;
        int r = N % thread_count;

        int my_count;
        int my_first_i;
        int my_last_i;

        if (my_rank < r)
        {
            my_count = q + 1;
            my_first_i = my_rank * (q + 1);
        }
        else
        {
            my_count = q;
            my_first_i = my_rank * q + r;
        }

        my_last_i = my_first_i + my_count;

        // Each thread calculates its partial sum
        for (int i = my_first_i; i < my_last_i; i++)
        {
            sum2 += a[i];
        }
    }

    finish = omp_get_wtime() - start;

    cout << "\n========== Book Solution ==========" << endl;
    cout << "Total Sum: " << sum2 << endl;
    cout << "Execution Time: " << finish << " seconds" << endl;

    return 0;
}





// N=24
// ========== My Solution ==========
// Total Sum: 300
// Execution Time: 0.000785554 seconds

// ========== Book Solution ==========
// Total Sum: 300
// Execution Time: 1.0831e-05 seconds