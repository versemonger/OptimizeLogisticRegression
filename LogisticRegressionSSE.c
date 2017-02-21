/*
 * All changes to code are copyright, 2017, Zhu Li, zhuli@unm.edu
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <x86intrin.h>

#define SAMPLE_NUMBER 1024
#define SAMPLE_ATTRIBUTE_NUMBER 32
#define INITIAL_WEIGHTS_RANGE 0.01
#define SAMPLE_VALUE_RANGE 50
#define CONVERGE_RATE 0.0001
#define ITERATION_NUMBER 6000
#define DATA_NUMBER 4
//#define DEBUG

/**
 *
 * @param n Length of the array.
 * @param range Range of the numbers in the array is [0, range].
 * @return An array filled with random numbers.
 */
float* generateRandomVectorFloat(int n, float range) {

  float* ptr = (float*)aligned_alloc(16, sizeof(float) * n);
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
    product = _mm_add_ps(_mm_mul_ps(xSSE[i], wSSE[i]), product);
  }

  float* dp = (float*)&product;
  float sum = 0;
  for (int i = 0; i < DATA_NUMBER; i++) {
    sum += *dp++;
  }
  return sum;
}


float logisticFunction(float* x, float* w, int n, float w0) {
  float sum = w0 + dotProduct(x, w, n);
  return 1 / (1 + exp(sum));
}


void updateDelta(__m128* deltaSSE, float** x, float* difference) {
  for (int i = 0; i < SAMPLE_NUMBER; i++) {
    __m128 *xiSSE = (__m128*)x[i];
    const __m128 multiplier = _mm_set1_ps(difference[i] * CONVERGE_RATE);
    for (int j = 0; j < SAMPLE_ATTRIBUTE_NUMBER / DATA_NUMBER; j++) {
      deltaSSE[j] = _mm_add_ps(_mm_mul_ps(xiSSE[j], multiplier), deltaSSE[j]);
    }
  }
}

void updateWeights(float* weights, float** x, float* y, float w0) {

  float* difference = (float*)aligned_alloc(16, sizeof(float) * SAMPLE_NUMBER);
  // Calculate the difference according to logistic regression update formula
  for (int i = 0; i < SAMPLE_NUMBER; i++) {
    difference[i] = logisticFunction(x[i], weights, SAMPLE_ATTRIBUTE_NUMBER, w0);
  }
  const __m128 minusOne = _mm_set1_ps(-1);
  __m128* diffSSE = (__m128*)difference;
  __m128* ySSE = (__m128*)y;
  for (int i = 0; i < SAMPLE_NUMBER / DATA_NUMBER; i++) {
    diffSSE[i] = _mm_add_ps(diffSSE[i], _mm_add_ps(ySSE[i], minusOne));
  }

  // Calculate the delta vector according to the update formula
  float* delta = (float*)aligned_alloc(16, sizeof(float) * SAMPLE_ATTRIBUTE_NUMBER);
  __m128* deltaSSE = (__m128*)delta;
  const __m128 zeros = _mm_set1_ps(0);
  for (int i = 0; i < SAMPLE_ATTRIBUTE_NUMBER / DATA_NUMBER; i++) {
    deltaSSE[i] = zeros;
  }

  updateDelta(deltaSSE, x, difference);

  // add delta to the original weights
  __m128* weightsSSE = (__m128*)weights;
  for (int i = 0; i <SAMPLE_ATTRIBUTE_NUMBER / DATA_NUMBER; i++) {
    weightsSSE[i] = _mm_add_ps(weightsSSE[i], deltaSSE[i]);
  }

  free(difference);
  free(delta);
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
  float y[SAMPLE_NUMBER];
  float* benchMarkWeights = (float*)aligned_alloc(16, SAMPLE_ATTRIBUTE_NUMBER * sizeof(float));
  float benchMarkWeight0 = rand() % 2 - 0.5;
  for (int i = 0; i < SAMPLE_ATTRIBUTE_NUMBER; i++) {
    benchMarkWeights[i] = rand() % 2 - 0.5;
  }
  for (int i = 0; i < SAMPLE_NUMBER; i++) {
    y[i] = logisticFunction(x[i], benchMarkWeights, SAMPLE_ATTRIBUTE_NUMBER, benchMarkWeight0) > 0.5 ? 0 : 1;
  }

  clock_t start = clock(), diff;
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
  for (int i = 0; i < SAMPLE_NUMBER; i++) {
    float predict = logisticFunction(x[i], weights, SAMPLE_ATTRIBUTE_NUMBER, w0) > 0.5 ? 0 : 1;
#ifdef DEBUG
    printf("y[%d]: %lf Predicted: %lf\n", i, y[i], predict);
#endif
    error += fabs(predict - y[i]);
  }
  printf("Average error:%lf\n", error / SAMPLE_NUMBER);

  diff = clock() - start;
  int msec = diff * 1000 / CLOCKS_PER_SEC;
  printf("Time taken: %d seconds %d milliseconds\n", msec / 1000, msec % 1000);
  for (int i = 0; i < SAMPLE_NUMBER; i++) {
    free(x[i]);
  }
  free(x);
  free(weights);
  free(benchMarkWeights);
  return 0;
}



