#include <mpi.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
/*
If your code includes mathematical functions (like exp, cos, etc.), you need to link to the mathematics 
library libm.so. This is done, just like for serial compiling, by adding -lm to the end of your compile command, 
that is,

mpicc mont-carlo_area_calculation.c -o mont-carlo_area_calculation -lm
*/
#define U(x) (sin(x) + 1)   //U(x) must be positive in my code 
#define OFFSET_U(a,b) (1 * (b - a)) //This is offset area added to our function

void save_results(
    int processes,
    int slices,
    long long total_points,
    double total_area,
    double elapsed_time
) {
    FILE *fp = fopen("results.csv", "a");

    if (fp == NULL) {
        perror("Error opening file");
        return;
    }

    fprintf(fp,
            "%d,%d,%lld,%.6f,%.6f\n",
            processes,
            slices,
            total_points,
            total_area,
            elapsed_time);

    fclose(fp);
}

double max_func(
    double a,
    double b
){
    double pi = acos(-1.0);

    double k_min = ceil((a - pi/2.0) / (2.0*pi));
    double k_max = floor((b - pi/2.0) / (2.0*pi));

    if (k_min <= k_max) {
        return 2;
    }else if (U(a) > U(b)) {
        return U(a);
    }else {
        return U(b);
    }
}

void mont_carlo(
    int rank_num,
    int comm_sz,
    double left_endpt,
    double right_endpt,
    double h,
    int local_n,
    int mont_carlo_n,
    double* total_area,
    double* local_area
){
    srand(time(NULL) + rank_num);
    // printf("rank: %d\n", rank_num);
    *local_area = 0; 
    for (int i = 0; i < local_n; i++) {
        int hit = 0;
        
        for (int j = 0; j < mont_carlo_n; j++) {
            double r_x_normalized = (double) rand() / RAND_MAX;
            double r_y_normalized = (double) rand() / RAND_MAX;
            
            double r_x = left_endpt + i * h + h * r_x_normalized;
            double r_y = max_func(left_endpt + i * h, left_endpt + (i + 1) * h ) * r_y_normalized;

            hit += (U(r_x) >= r_y) ? 1 : 0;
        }
        *local_area += max_func(left_endpt + i * h, left_endpt + (i + 1) * h ) * h * ((double)hit / mont_carlo_n);;
    }
    printf("rank: %d, left_endpt: %lf, right_endpt: %lf, local area: %lf\n", rank_num, left_endpt, right_endpt, *local_area);
    MPI_Reduce(local_area, total_area, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    

}

void get_input(
    int rank_num,
    int comm_sz,
    double* a_p,
    double* b_p,
    int* n_p,
    int* mont_carlo_n,
    double* h,
    time_t* start_time
){
    if (rank_num == 0){
        printf("Enter a, b, n and mont_carlo_n\n");
        scanf("%lf %lf %d %d", a_p, b_p, n_p, mont_carlo_n);
    }
    *h = (*b_p - *a_p) / *n_p;

    MPI_Bcast(a_p, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(b_p, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(n_p, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(mont_carlo_n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank_num == 0) {
        time(start_time);
    }
}


void distribute_local_values(
    int rank_num,
    int comm_size,
    double a,
    double b,
    double h,
    double* local_a,
    double* local_b,
    int n,
    int* local_n
){
    /*calculate local_n*/
    *local_n = n / comm_size;
    int remainder = n % comm_size;
    *local_n += (rank_num < remainder) ? 1 : 0;

    /*calculate local_a and local_b*/
    *local_a = a + (rank_num * (*local_n) + ((rank_num >= remainder) ? remainder:0)) * h;
    *local_b = *local_a + (*local_n) * h;

    printf("rank: %d ,local a: %lf, local b: %lf, h: %lf, local n: %d\n",rank_num, *local_a, *local_b, h, *local_n);
}
    
int main(void){

    int comm_size, rank_num, n, local_n, mont_carlo_n;
    double a, b, h, local_a, local_b, total_area, local_area;
    time_t start_time, end_time;


    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_num);

    get_input(rank_num, comm_size, &a, &b, &n, &mont_carlo_n, &h, &start_time);

    distribute_local_values(rank_num, comm_size, a, b, h, &local_a, &local_b, n, &local_n);

    mont_carlo(rank_num, comm_size, local_a, local_b, h, local_n, mont_carlo_n, &total_area, &local_area);

    if(rank_num == 0){
        total_area -= OFFSET_U(a,b);

        time(&end_time);
        double elapsed = difftime(end_time, start_time) * 1000000;

        FILE *fp = fopen("results.csv", "r");
        if (fp == NULL) {
            fp = fopen("results.csv", "w");
            fprintf(fp,
                "Processes,Slices,Total Points,Total Area,Elapsed Time (us)\n");
            fclose(fp);    
        } else {
            fclose(fp);    
        }


        printf("┌──────────────────────────────────┐\n");
        printf("│ Results                          │\n");
        printf("├──────────────────────────────────┤\n");
        printf("│ Processes        : %-16d │\n", comm_size);
        printf("│ a                : %-16.4lf │\n", a);
        printf("│ b                : %-16.4lf │\n", b);
        printf("│ Number of slices : %-16d │\n", n);
        printf("│ Total points     : %-16d │\n", mont_carlo_n);
        printf("│ Total Area       : %-16.6lf │\n", total_area);
        printf("│ Elapsed Time     : %-11.3f Mus │\n", elapsed);
        printf("└──────────────────────────────────┘\n");
        
        save_results(
            comm_size,
            n,
            mont_carlo_n,
            total_area,
            elapsed
        );
    }

    MPI_Finalize();
    return 0;
}