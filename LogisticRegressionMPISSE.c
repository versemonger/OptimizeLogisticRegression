/*
 * All changes to code are copyright, 2017, Zhu Li, zhuli@unm.edu
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <x86intrin.h>
#include <mpi/mpi.h>
#include <sys/time.h>

#define SAMPLE_NUMBER 1024 * 4
#define SAMPLE_ATTRIBUTE_NUMBER 32
#define INITIAL_WEIGHTS_RANGE 0.01
#define SAMPLE_VALUE_RANGE 50
#define CONVERGE_RATE 0.0001
#define ITERATION_NUMBER 6000 * 2
#define DATA_NUMBER 4
#define BYTE_NUMBER 16
#define MICROSEC_IN_SEC 1000000
//#define DEBUG


int comm_sz; //Number of Processes
int my_rank; //My process rank

/**
 *
 * @param n Length of the array.
 * @param range Range of the numbers in the array is [0, range].
 * @return An array filled with random numbers.
 */
float* generateRandomVectorFloat(int n, float range) {

  float* ptr = (float*)aligned_alloc(BYTE_NUMBER, sizeof(float) * n);
  if (ptr != NULL) {
    for (int i = 0; i < n; i++) {
      ptr[i] =  (range * rand() / RAND_MAX) - range / 2;
    }
  }
  return ptr;
}

/**
 *  return dot product of vector x and w.
 */
float dotProduct(float* x, float* w, int n) {
  __m128 product = _mm_set1_ps(0);
  __m128* xSSE = (__m128*)x;
  __m128* wSSE = (__m128*)w;
  for (int i = 0; i < n / DATA_NUMBER; i++) {
    product = _mm_add_ps(product, _mm_mul_ps(xSSE[i], wSSE[i]));
    //product = _mm256_fmadd_ps(xSSE[i], wSSE[i], product);
  }

  float* dp = (float*)&product;
  float sum = 0;
  for (int i = 0; i < DATA_NUMBER; i++) {
    sum += *dp++;
  }
  return sum;
}


float logisticFunction(float* x, float* w, int n, float w0) {
  return 1 / (1 + exp(w0 + dotProduct(x, w, n)));
}

void updateDelta(float **x, float *difference, __m128* weightsSSE) {
  float* weights = (float*) weightsSSE;
  float converge_rate = CONVERGE_RATE;
  int splitSize = SAMPLE_ATTRIBUTE_NUMBER / DATA_NUMBER / comm_sz;
  int start = splitSize * my_rank;
  int end = start + splitSize;
  for (int j = start; j < end; j++) {
    for (int i = 0; i < SAMPLE_NUMBER; i++) {
      __m128 *xiSSE = (__m128*)x[i];
      const __m128 multiplier = _mm_set1_ps(difference[i] * converge_rate);
      weightsSSE[j] = _mm_add_ps(weightsSSE[j], _mm_mul_ps(xiSSE[j], multiplier));
      //weightsSSE[j] = _mm256_fmadd_ps(xiSSE[j], multiplier, weightsSSE[j]);
    }
  }
  MPI_Allgather(weights + start * DATA_NUMBER, splitSize * DATA_NUMBER, MPI_FLOAT, weights, splitSize * DATA_NUMBER, MPI_FLOAT, MPI_COMM_WORLD);
}

void updateWeights(float* weights, float** x, float* y, float w0) {

  float *difference = (float *) aligned_alloc(BYTE_NUMBER, sizeof(float) * SAMPLE_NUMBER);
  const __m128 minusOne = _mm_set1_ps(-1);
  __m128 *diffSSE = (__m128 *) difference;
  __m128 *ySSE = (__m128 *) y;
  __m128* weightsSSE = (__m128*)weights;
  // Divide workload among different processes
  int splitSize = SAMPLE_NUMBER / comm_sz;
  int start = my_rank * splitSize;
  int end = start + splitSize;
  // Calculate the difference according to logistic regression update formula
  for (int i = start; i < end; i++) {
    difference[i] = logisticFunction(x[i], weights, SAMPLE_ATTRIBUTE_NUMBER, w0);
  }
  for (int i = start / DATA_NUMBER; i < end / DATA_NUMBER; i++) {
    diffSSE[i] = _mm_add_ps(diffSSE[i], _mm_add_ps(ySSE[i], minusOne));
  }
  MPI_Allgather(difference + start, splitSize, MPI_FLOAT, difference, splitSize, MPI_FLOAT, MPI_COMM_WORLD);
  // update and add delta to the original weights
  // function name is not changed for consistency
  updateDelta(x, difference, weightsSSE);
  free(difference);
}



int main() {
  srand(time(NULL));
  // initialize the weights randomly
  float w0 = (INITIAL_WEIGHTS_RANGE * rand() / RAND_MAX) - INITIAL_WEIGHTS_RANGE / 2;
  float* weights = generateRandomVectorFloat(SAMPLE_ATTRIBUTE_NUMBER, INITIAL_WEIGHTS_RANGE);
  // TODO: load real data into x and y;
  // Generate random data for x

  float** x = (float**)malloc(SAMPLE_NUMBER * sizeof(float*));
  for (int i = 0; i < SAMPLE_NUMBER; i++) {
    x[i] = generateRandomVectorFloat(SAMPLE_ATTRIBUTE_NUMBER, SAMPLE_VALUE_RANGE);
  }

  // Set all benchmark weights as 0.5 or -0.5 randomly and generate the corresponding labels.
  // So we could test the effectiveness of the program according to whether
  // the program could predict the labels generated with benchmark weights
  float* y = (float*)aligned_alloc(BYTE_NUMBER, SAMPLE_NUMBER * sizeof(float));
  float* benchMarkWeights = (float*)aligned_alloc(BYTE_NUMBER, SAMPLE_ATTRIBUTE_NUMBER * sizeof(float));
  float benchMarkWeight0 = rand() % 2 - 0.5;
  for (int i = 0; i < SAMPLE_ATTRIBUTE_NUMBER; i++) {
    benchMarkWeights[i] = rand() % 2 - 0.5;
  }
  for (int i = 0; i < SAMPLE_NUMBER; i++) {
    y[i] = logisticFunction(x[i], benchMarkWeights, SAMPLE_ATTRIBUTE_NUMBER, benchMarkWeight0) > 0.5 ? 0 : 1;
  }
  struct timeval tv;
  gettimeofday(&tv, NULL);
  long start = tv.tv_usec + tv.tv_sec * MICROSEC_IN_SEC;
  //clock_t start = clock(), diff;
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  for (int i = 0; i < ITERATION_NUMBER; i++) {
    updateWeights(weights, x, y, w0);
  }

#ifdef DEBUG
  for (int i = 0; i < SAMPLE_ATTRIBUTE_NUMBER; i++) {
    printf("Benchmark weight: %lf Estimated weight:%lf\n", benchMarkWeights[i], weights[i]);
  }
#endif
  // Predict the labels with weights estimated with logistic regression.
  float error = 0;
  // Split workload among several processes
  int splitSize = SAMPLE_NUMBER / comm_sz;
  int splitStart = splitSize * my_rank;
  int splitEnd = splitStart + splitSize;
  for (int i = splitStart; i < splitEnd; i++) {
    float predict = logisticFunction(x[i], weights, SAMPLE_ATTRIBUTE_NUMBER, w0) > 0.5 ? 0 : 1;
#ifdef DEBUG
    printf("y[%d]: %lf Predicted: %lf\n", i, y[i], predict);
#endif
    error += fabs(predict - y[i]);
  }
  float totalError = 0;
  MPI_Reduce(&error, &totalError, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  if (my_rank == 0) {
    printf("Average error:%lf\n", error / SAMPLE_NUMBER);
    //int diff = gettimeofday() - start;
    //diff = clock() - start;
    gettimeofday(&tv, NULL);
    long diff = (tv.tv_sec * MICROSEC_IN_SEC + tv.tv_usec - start) / 1000;
    printf("Time taken: %ld seconds %ld milliseconds\n", diff / 1000, diff % 1000);
  }
  MPI_Finalize();
  for (int i = 0; i < SAMPLE_NUMBER; i++) {
    free(x[i]);
  }
  free(x);
  free(y);
  free(weights);
  free(benchMarkWeights);
  return 0;
}

