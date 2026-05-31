#include <mpi.h>
#include <stdio.h>
#include <math.h>
/*
If your code includes mathematical functions (like exp, cos, etc.), you need to link to the mathematics 
library libm.so. This is done, just like for serial compiling, by adding -lm to the end of your compile command, 
that is,

mpicc mont-carlo_area_calculation.c -o mont-carlo_area_calculation -lm
*/
#define U(x) (sin(x))


int mont_carlo(
    double left_endpt,
    double right_endpt
);

void get_input(
    int rank_num,
    int comm_sz,
    double* a_p,
    double* b_p
){
    if (rank_num == 0){
        printf("Enter a, b\n");
        scanf("%lf %lf", a_p, b_p);
    }
    MPI_Bcast(a_p, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(b_p, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void distribute_data(
    int rank_num,
    int comm_size,
    double* a_p,
    double* b_p,
    double* h,
    double* local_a,
    double* local_b
){
    *local_a = *a_p + rank_num * (*h);
    *local_b = *local_a + (*h);
}
    
int main(void){
    int comm_size, rank_num, n;
    double a, b, h, local_a, local_b;

    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_num);

    get_input(rank_num, comm_size, &a, &b);
    h = (b - a) / comm_size;

    distribute_data(rank_num, comm_size, &a, &b, &h, &local_a, &local_b);


    printf("rank: %d, local_a: %fl, local_b: %fl \n", rank_num, local_a, local_b);




    MPI_Finalize();
    return 0;
}