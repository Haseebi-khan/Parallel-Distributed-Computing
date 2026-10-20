// noneo@noneo-Precision-5530:~/Codes/PDC/Parallel-Distributed-Computing$ ./sum
// Sum = 276
// Threads used = 12
// Parallel Execution time = 3346 microseconds
// Sum = 276
// Threads used = 12
// Series Execution time = 0 microseconds
// noneo@noneo-Precision-5530:~/Codes/PDC/Parallel-Distributed-Computing$ g++ -fopenmp sum.cpp -o sum
// noneo@noneo-Precision-5530:~/Codes/PDC/Parallel-Distributed-Computing$ ./sum
// Sum = 4999999850000001
// Threads used = 12
// Parallel Execution time = 34160 microseconds
// Sum = 4999999850000001
// Threads used = 12
// Series Execution time = 203948 microseconds
// noneo@noneo-Precision-5530:~/Codes/PDC/Parallel-Distributed-Computing$ 

#include <iostream>
#include <omp.h>
#include <time.h>
#include <chrono>

using namespace std;
using namespace chrono;

int main() {
    long long int n = 99999999;
    long long int sum = 0;

    auto start = steady_clock::now();
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum += i;
    }
    auto end = steady_clock::now();
    auto elapsed = chrono::duration_cast<chrono::microseconds>(end - start);

    cout << "Sum = " << sum << endl;
    cout << "Threads used = " << omp_get_max_threads() << endl;
    cout << "Parallel Execution time = " << (elapsed.count()) << " microseconds" << endl;


    sum = 0;

    auto start1 = steady_clock::now();
    for (int i = 0; i < n; i++)
    {
        sum += i;
    }
    auto end1 = end = steady_clock::now();
    auto elapsed1 = chrono::duration_cast<chrono::microseconds>(end1 - start1);

    cout << "Sum = " << sum << endl;
    cout << "Threads used = " << omp_get_max_threads() << endl;
    cout << "Series Execution time = " << (elapsed1.count()) << " microseconds" << endl;
    
    return 0;
}
