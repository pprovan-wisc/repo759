#include <iostream>//libraries with needed functions
#include <cstdio>
#include <cstdlib>

int main(int argc, char* argv[]) {
    //make sure that a command line argument is provided
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " N" << std::endl;
        return 1;
    }

    //covert to an integer
    int N = std::atoi(argv[1]);

    //pritning from 0 to N using printf
    for (int i = 0; i <= N; ++i) {
        printf("%d", i);
        if (i != N) printf(" "); //space between numbers
    }
    printf("\n"); //newline

    //printing N to 0 using std::cout
    for (int i = N; i >= 0; --i) {
        std::cout << i;
        if (i != 0) std::cout << " "; //space between numbers
    }
    std::cout << std::endl; //newline

    return 0;
}
