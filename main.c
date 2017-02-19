#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


#define SAMPLE_NUMBER 1000
#define SAMPLE_ATTRIBUTE_NUMBER 20
#define INITIAL_WEIGHTS_RANGE 0.01
#define SAMPLE_VALUE_RANGE 50
#define CONVERGE_RATE 0.0001
#define ITERATION_NUMBER 1000
// #define DEBUG

/**
 *
 * @param n Length of the array.
 * @param range Range of the numbers in the array is [0, range].
 * @return An array filled with random numbers.
 */
double* generateRandomVectorDouble(int n, double range) {

  double* ptr = (double*)malloc(sizeof(double) * n);
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
double* generateRandomVectorBoolean(int n) {
  double* ptr = (double*)malloc(sizeof(double) * n);
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
double dotProduct(double* x, double* w, int n) {
  double dotProdcut = 0;
  for (int i = 0; i < n; i++) {
    dotProdcut += x[i] * w[i];
  }
}


double logisticFunction(double* x, double* w, int n) {
  double sum = *w + dotProduct(x, w + 1, n);
  return 1 / (1 + exp(sum));
}


void updateWeights(double* weights, double** x, double* y) {
  double difference[SAMPLE_NUMBER];
  // Calculate the difference according to logistic regression update formula
  for (int i = 0; i < SAMPLE_NUMBER; i++) {
      difference[i] = y[i] + logisticFunction(x[i], weights, SAMPLE_ATTRIBUTE_NUMBER) - 1;
  }
  // Calculate the delta vector according to the update formula
  double delta[SAMPLE_ATTRIBUTE_NUMBER] = {0};
  for (int i = 0; i < SAMPLE_NUMBER; i++) {
    for (int j = 0; j < SAMPLE_ATTRIBUTE_NUMBER; j++) {
      delta[j] += x[i][j] * CONVERGE_RATE * difference[i];
    }
  }
  // add delta to the original weights
  for (int i = 1; i <= SAMPLE_ATTRIBUTE_NUMBER; i++) {
    weights[i] += delta[i - 1];
  }
}


int main() {
  srand(time(NULL));
  // initialize the weights randomly
  double* weights = generateRandomVectorDouble(SAMPLE_ATTRIBUTE_NUMBER + 1, INITIAL_WEIGHTS_RANGE);
  // TODO: load real data into x and y;
  // Generate random data for x
  double** x = (double**)malloc(SAMPLE_NUMBER * sizeof(double*));
  for (int i = 0; i < SAMPLE_NUMBER; i++) {
    x[i] = generateRandomVectorDouble(SAMPLE_ATTRIBUTE_NUMBER, SAMPLE_VALUE_RANGE);
  }

  // Set all benchmark weights as 0.5 or -0.5 randomly and generate the corresponding labels.
  // So we could test the effectiveness of the program according to whether
  // the program could predict the labels generated with benchmark weights
  double y[SAMPLE_NUMBER];
  double benchMarkWeights[SAMPLE_ATTRIBUTE_NUMBER + 1];
  for (int i = 0; i < SAMPLE_ATTRIBUTE_NUMBER + 1; i++) {
    benchMarkWeights[i] = rand() % 2 - 0.5;
  }
  for (int i = 0; i < SAMPLE_NUMBER; i++) {
    y[i] = logisticFunction(x[i], benchMarkWeights, SAMPLE_ATTRIBUTE_NUMBER) > 0.5 ? 0 : 1;
  }
  printf("\n");

  for (int i = 0; i < ITERATION_NUMBER; i++) {
    updateWeights(weights, x, y);
  }

#ifdef DEBUG
  for (int i = 0; i < SAMPLE_ATTRIBUTE_NUMBER; i++) {
    printf("Benchmark weight: %lf Estimated weight:%lf\n", benchMarkWeights[i], weights[i]);
  }
#endif
  // Predict the labels with weights estimated with logistic regression.
  double error = 0;
  for (int i = 0; i < SAMPLE_NUMBER; i++) {
    double predict = logisticFunction(x[i], weights, SAMPLE_ATTRIBUTE_NUMBER) > 0.5 ? 0 : 1;
#ifdef DEBUG
    printf("y[%d]: %lf Predicted: %lf\n", i, y[i], predict);
#endif
    error += fabs(predict - y[i]);
  }
  printf("Average error:%lf\n", error / SAMPLE_NUMBER);
  return 0;
}



