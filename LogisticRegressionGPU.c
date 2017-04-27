/*
 * All changes to code are copyright, 2017, Zhu Li, zhuli@unm.edu
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <openacc.h>
#define SAMPLE_NUMBER 1024 * 4
#define SAMPLE_ATTRIBUTE_NUMBER 32 * 2
#define INITIAL_WEIGHTS_RANGE 0.01
#define SAMPLE_VALUE_RANGE 50
#define CONVERGE_RATE 0.0001
#define ITERATION_NUMBER 6000 * 2
#define MICROSEC_IN_SEC 1000000

//#define DEBUG

/**
 *
 * @param n Length of the array.
 * @param range Range of the numbers in the array is [0, range].
 * @return An array filled with random numbers.
 */
float* generateRandomVectorFloat(int n, float range) {
  float* ptr = (float*)malloc(sizeof(float) * n);
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
#pragma acc routine
float dotProduct(float* x, float* w, int n) {
  float sum = 0;
  for (int i = 0; i < n; i++) {
    sum += x[i] * w[i];
  }
  return sum;
}

#pragma acc routine
float logisticFunction(float* x, float* w, int n, float w0) {
  float sum = w0 + dotProduct(x, w, n);
  return 1 / (1 + exp(sum));
}

#pragma acc routine
void updateWeights(float* weights, float** x, float* y, float w0, float* difference) {
#pragma acc loop gang vector
  for (int i = 0; i < SAMPLE_NUMBER; i++) {
    difference[i] = logisticFunction(x[i], weights, SAMPLE_ATTRIBUTE_NUMBER, w0) + y[i] - 1;
  }
#pragma acc loop gang vector
  for (int j = 0; j < SAMPLE_ATTRIBUTE_NUMBER; j++) {
    for (int i = 0; i < SAMPLE_NUMBER; i++) {
      weights[j] += x[i][j] * difference[i] * CONVERGE_RATE;
    }
  }
}

int main() {
  acc_set_device_num(5, acc_device_nvidia);
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
  float* y = (float*)malloc(SAMPLE_NUMBER * sizeof(float));
  float* benchMarkWeights = (float*)malloc(SAMPLE_ATTRIBUTE_NUMBER * sizeof(float));
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
  float *difference = (float *) malloc(sizeof(float) * SAMPLE_NUMBER);
#pragma acc enter data copyin(weights[0:SAMPLE_ATTRIBUTE_NUMBER], x[0:SAMPLE_NUMBER][0:SAMPLE_ATTRIBUTE_NUMBER], y[0:SAMPLE_NUMBER], difference[0:SAMPLE_NUMBER])
#pragma acc parallel vector_length(64) present(weights[0:SAMPLE_ATTRIBUTE_NUMBER], x[0:SAMPLE_NUMBER][0:SAMPLE_ATTRIBUTE_NUMBER], y[0:SAMPLE_NUMBER], difference[0:SAMPLE_NUMBER])
  {
    for (int i = 0; i < ITERATION_NUMBER; i++)
    {
      updateWeights(weights, x, y, w0, difference);
    }
  }
#pragma acc exit data copyout(weights[0:SAMPLE_ATTRIBUTE_NUMBER])
  free(difference);


#ifdef DEBUG
  for (int i = 0; i < SAMPLE_ATTRIBUTE_NUMBER; i++) {
    printf("Benchmark weight: %lf Estimated weight:%lf\n", benchMarkWeights[i], weights[i]);
  }
#endif
  // Predict the labels with weights estimated with logistic regression.
  float error = 0;
  for (int i = 0; i < SAMPLE_NUMBER; i++) {
    float predict = logisticFunction(x[i], weights, SAMPLE_ATTRIBUTE_NUMBER, w0) > 0.5 ? 0 : 1;
#ifdef DEBUG
    printf("y[%d]: %lf Predicted: %lf\n", i, y[i], predict);
#endif
    error += fabs(predict - y[i]);
  }
  printf("Average error:%lf\n", error / SAMPLE_NUMBER);
  //int diff = gettimeofday() - start;
  //diff = clock() - start;
  gettimeofday(&tv, NULL);
  long diff = (tv.tv_sec * MICROSEC_IN_SEC + tv.tv_usec - start) / 1000;
  printf("Time taken: %ld seconds %ld milliseconds\n", diff / 1000, diff % 1000);
  for (int i = 0; i < SAMPLE_NUMBER; i++) {
    free(x[i]);
  }
  free(x);
  free(y);
  free(weights);
  free(benchMarkWeights);
  return 0;
}



