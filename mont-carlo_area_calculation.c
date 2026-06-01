#include <ctime>
#include <mpi.h>
#include <stdio.h>
#include <time.h>
/*
If your code includes mathematical functions (like exp, cos, etc.), you need to link to the mathematics 
library libm.so. This is done, just like for serial compiling, by adding -lm to the end of your compile command, 
that is,

mpicc mont-carlo_area_calculation.c -o mont-carlo_area_calculation -lm
*/
#define U(x) (sin(x) + 1)   //U(x) must be positive in my code 


int mont_carlo(
    double left_endpt,
    double right_endpt
);

void get_input(
    int rank_num,
    int comm_sz,
    double* a_p,
    double* b_p,
    int* n_p,
    double* h
){
    
    if (rank_num == 0){
        printf("Enter a, b\n");
        scanf("%lf %lf %d", a_p, b_p, n_p);
    }
    *h = (*b_p - *a_p) / *n_p;

    MPI_Bcast(a_p, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(b_p, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(n_p, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void distribute_data(
    int rank_num,
    int comm_size,
    double* a_p,
    double* b_p,
    int* n_p,
    double* h_p,
    double* local_a,
    double* local_b,
    int* local_n,
    int* send_counts,
    int* displacements
){
    int sub_area;
    
    *local_n = *n_p / comm_size;
    int remainder = *n_p % comm_size;
    *local_n += (rank_num < remainder) ? 1 : 0;

    if(rank_num == 0){
        int offset = 0;
        for (int i = 0; i< comm_size; i++) {
            sub_area = *n_p / comm_size + ((i < remainder) ? 1 : 0);
            send_counts[i] = sub_area;
            displacements[i] = offset;
            offset += sub_area;
        }
    } 

    MPI_Bcast(send_counts, *n_p, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displacements, *n_p, MPI_INT, 0, MPI_COMM_WORLD);

}
    
int main(void){
    int comm_size, rank_num, n, local_n;
    double a, b, h, local_a, local_b;
    int send_counts[comm_size], displacements[comm_size];

    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_num);

    get_input(rank_num, comm_size, &a, &b, &n, &h);

    distribute_data(rank_num, comm_size, &a, &b, &n, &h, &local_a, &local_b, &local_n, send_counts, displacements);

    


    MPI_Finalize();
    return 0;
}