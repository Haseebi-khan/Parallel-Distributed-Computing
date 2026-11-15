#include <iostream>
#include <omp.h>

using namespace std;

int main(int argc, char *argv[]){

    int thread_count = omp_get_max_threads();
    cout << "Check thread_count: " << thread_count << endl;

    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        int total = omp_get_num_threads();
        #pragma omp critical
        cout << "Hello from thread "<< id << "  of  "<< total <<endl;
    }

    return 0;
}