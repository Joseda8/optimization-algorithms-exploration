# Optimization Algorithm Benchmarking Project

This Python project is designed for benchmarking optimization algorithms, specifically focusing on Particle Swarm Optimization (PSO) but with the ability to be extended as needed. The project includes implementations of the PSO algorithm, various benchmark functions, and utility modules for logging and data writing. The results of the optimization runs are stored in a CSV file.

## Running the Benchmark

To run the benchmark experiments, execute the `main.py` file. The benchmark tests PSO on a set of benchmark functions and records the results, including the best position and value.

```bash
python3 -m src.main
```

## Benchmarked Functions

The project tests the PSO algorithm on the following benchmark functions:

1. Ackley.
2. Branin Rcos.
3. Goldstein Price.
4. Griewank.
5. Levy.
6. Michalewicz.
7. Rastrigin.
8. Rosenbrock.
9. Six-Hump Camelback.
10. Sphere.

## Results

The results of the benchmark experiments are stored in a CSV file located at `results/optimization_results.csv`. 
