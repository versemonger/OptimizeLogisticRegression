#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


#define SAMPLE_NUMBER 1024
#define SAMPLE_ATTRIBUTE_NUMBER 32
#define INITIAL_WEIGHTS_RANGE 0.01
#define SAMPLE_VALUE_RANGE 50
#define CONVERGE_RATE 0.0001
#define ITERATION_NUMBER 6000
#define DATA_NUMBER 4
#define DEBUG

/**
 *
 * @param n Length of the array.
 * @param range Range of the numbers in the array is [0, range].
 * @return An array filled with random numbers.
 */
float* generateRandomVectorfloat(int n, float range) {

  float* ptr = (float*)malloc(sizeof(float) * n);
  if (ptr != NULL) {
    for (int i = 0; i < n; i++) {
      ptr[i] =  (range * rand() / RAND_MAX) - range / 2;
    }
  }
  return ptr;
}

/**
 *
 * @param n length of the vector; number of samples in this scenario.
 * @return a vector with values of randomly 1 or 0.
 */
float* generateRandomVectorBoolean(int n) {
  float* ptr = (float*)malloc(sizeof(float) * n);
  if (ptr != NULL) {
    for (int i = 0; i < n; i++) {
      ptr[i] = rand() % 2;
    }
  }
  return ptr;
}


/**
 *  return dot product of vector x and w.
 */
float dotProduct(float* x, float* w, int n) {
  float dotProduct = 0;
  for (int i = 0; i < n; i++) {
    dotProduct += x[i] * w[i];
  }
}


float logisticFunction(float* x, float* w, int n) {
  float sum = *w + dotProduct(x, w + 1, n);
  return 1 / (1 + exp(sum));
}


void updateWeights(float* weights, float** x, float* y) {
  float* difference = (float*)malloc(sizeof(float) * SAMPLE_NUMBER);
  // Calculate the difference according to logistic regression update formula
  for (int i = 0; i < SAMPLE_NUMBER; i++) {
    difference[i] = y[i] + logisticFunction(x[i], weights, SAMPLE_ATTRIBUTE_NUMBER) - 1;
  }
  // Calculate the delta vector according to the update formula
  float* delta = (float*)calloc(SAMPLE_ATTRIBUTE_NUMBER, sizeof(float));
  
  for (int i = 0; i < SAMPLE_NUMBER; i++) {
    for (int j = 0; j < SAMPLE_ATTRIBUTE_NUMBER; j++) {
      delta[j] += x[i][j] * CONVERGE_RATE * difference[i];
    }
  }
  // add delta to the original weights
  for (int i = 1; i <= SAMPLE_ATTRIBUTE_NUMBER; i++) {
    weights[i] += delta[i - 1];
  }
  free(delta);
}


int main() {
  srand(time(NULL));
  clock_t start = clock(), diff;
  // initialize the weights randomly
  float* weights = generateRandomVectorfloat(SAMPLE_ATTRIBUTE_NUMBER + 1, INITIAL_WEIGHTS_RANGE);
  // TODO: load real data into x and y;
  // Generate random data for x
  float** x = (float**)malloc(SAMPLE_NUMBER * sizeof(float*));
  for (int i = 0; i < SAMPLE_NUMBER; i++) {
    x[i] = generateRandomVectorfloat(SAMPLE_ATTRIBUTE_NUMBER, SAMPLE_VALUE_RANGE);
  }

  // Set all benchmark weights as 0.5 or -0.5 randomly and generate the corresponding labels.
  // So we could test the effectiveness of the program according to whether
  // the program could predict the labels generated with benchmark weights
  float y[SAMPLE_NUMBER];
  float benchMarkWeights[SAMPLE_ATTRIBUTE_NUMBER + 1];
  for (int i = 0; i < SAMPLE_ATTRIBUTE_NUMBER + 1; i++) {
    benchMarkWeights[i] = rand() % 2 - 0.5;
  }
  for (int i = 0; i < SAMPLE_NUMBER; i++) {
    y[i] = logisticFunction(x[i], benchMarkWeights, SAMPLE_ATTRIBUTE_NUMBER) > 0.5 ? 0 : 1;
  }

  for (int i = 0; i < ITERATION_NUMBER; i++) {
    updateWeights(weights, x, y);
  }

#ifdef DEBUG
  for (int i = 0; i < SAMPLE_ATTRIBUTE_NUMBER; i++) {
    printf("Benchmark weight: %lf Estimated weight:%lf\n", benchMarkWeights[i], weights[i]);
  }
#endif
  // Predict the labels with weights estimated with logistic regression.
  float error = 0;
  for (int i = 0; i < SAMPLE_NUMBER; i++) {
    float predict = logisticFunction(x[i], weights, SAMPLE_ATTRIBUTE_NUMBER) > 0.5 ? 0 : 1;
#ifdef DEBUG
    printf("y[%d]: %lf Predicted: %lf\n", i, y[i], predict);
#endif
    error += fabs(predict - y[i]);
  }
  printf("Average error:%lf\n", error / SAMPLE_NUMBER);

  diff = clock() - start;
  int msec = diff * 1000 / CLOCKS_PER_SEC;
  printf("Time taken: %d seconds %d milliseconds\n", msec / 1000, msec % 1000);
  free(weights);
  free(x);
  return 0;
}



