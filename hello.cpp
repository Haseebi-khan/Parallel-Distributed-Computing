// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc$ g++ -g -Wall -fopenmp -o hello hello.cpp
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc$ ./hello 4
// Hello from thread: 1
// Hello from thread: 3
// Hello from thread: 2
// Hello from thread: 0
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc$ g++ -g -Wall -o hello hello.cpp
// hello.cpp: In function ‘int main(int, char**)’:
// hello.cpp:23:9: warning: variable ‘thread_count’ set but not used [-Wunused-but-set-variable]
//    23 |     int thread_count = 0;
//       |         ^~~~~~~~~~~~
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc$ g++ -g -Wall -o hello hello.cpp
// hello.cpp: In function ‘int main(int, char**)’:
// hello.cpp:27:13: warning: unused variable ‘thread_count’ [-Wunused-variable]
//    27 |         int thread_count = strtol(argv[1], NULL, 10);
//       |             ^~~~~~~~~~~~
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc$ g++ -g -Wall -o hello hello.cpp
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc$ ./hello 4
// Hello from the main thread!
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc$ 

#include <iostream>
#include <cstdlib>

// #ifdef _OPENMP
#include <omp.h>
#include <chrono>
using namespace std::chrono;
// #endif

using namespace std;

void hello_Fun(void)
{
    // #ifdef _OPENMP
        cout << "Hello from thread: " << omp_get_thread_num() << endl;
    // #else
    //     cout << "Hello from the main thread!" << endl;
    // #endif
}

int main(int argc, char *argv[])
{
    // int thread_count = 0;

    // else
    // {
        //     thread_count = 1;
        // }
        
    // #ifdef _OPENMP
        if (argc > 1)
        {
            int thread_count = strtol(argv[1], NULL, 10);
        }
        #pragma omp parallel num_threads(thread_count)
        hello_Fun();

    // #else
    //     hello_Fun();
    // #endif
    
    return 0;
}

// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc$ g++ -g -Wall -o hello hello.cpp
// hello.cpp:59: warning: ignoring ‘#pragma omp parallel’ [-Wunknown-pragmas]
//    59 |         #pragma omp parallel num_threads(thread_count)
//       | 
// hello.cpp: In function ‘int main(int, char**)’:
// hello.cpp:57:17: warning: unused variable ‘thread_count’ [-Wunused-variable]
//    57 |             int thread_count = strtol(argv[1], NULL, 10);
//       |                 ^~~~~~~~~~~~
// /usr/bin/ld: /tmp/ccmmfPeJ.o: in function `hello_Fun()':
// /home/noneo/A Semester/PDC/Code/pdc/hello.cpp:39:(.text+0x2a): undefined reference to `omp_get_thread_num'
// collect2: error: ld returned 1 exit status
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc$ g++ -g -o hello hello.cpp
// /usr/bin/ld: /tmp/ccxKF4zX.o: in function `hello_Fun()':
// /home/noneo/A Semester/PDC/Code/pdc/hello.cpp:39:(.text+0x2a): undefined reference to `omp_get_thread_num'
// collect2: error: ld returned 1 exit status
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc$ ^C