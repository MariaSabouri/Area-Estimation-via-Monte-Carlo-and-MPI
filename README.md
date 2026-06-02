# Area-Estimation-via-Monte-Carlo-and-MPI
Parallel Monte Carlo integration using MPI for numerical approximation of integrals arising from differential equation problems.

This project implements a parallel Monte Carlo integration algorithm using MPI (Message Passing Interface) in C.

In many scientific and engineering applications, solving differential equations and evaluating integrals are fundamental tasks. While classical numerical methods such as Euler and Runge-Kutta rely on domain discretization, some problems can be reformulated into integral expressions.

This project investigates the use of the Monte Carlo method, a stochastic sampling-based approach, to approximate such integrals. Since random samples can be generated and evaluated independently, the algorithm is naturally suited for parallel execution.

The implementation distributes the sampling workload among multiple MPI processes, combines the local results, and computes a global estimate of the integral. The project serves as an introduction to distributed-memory parallel computing, numerical integration, and Monte Carlo methods.
