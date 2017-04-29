/*
 * All changes to code are copyright, 2017, Zhu Li, zhuli@unm.edu
 */

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <cuda.h>
#define SAMPLE_NUMBER 2048
#define SAMPLE_ATTRIBUTE_NUMBER 32
#define INITIAL_WEIGHTS_RANGE 0.01
#define SAMPLE_VALUE_RANGE 50
#define CONVERGE_RATE 0.0001
#define ITERATION_NUMBER 10000
#define MICROSEC_IN_SEC 1000000

#define DEBUG

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
__host__ __device__ float dotProduct(float* x, float* w, int n) {
  float sum = 0;
  for (int i = 0; i < n; i++) {
    sum += x[i] * w[i];
  }
  return sum;
}


__host__ __device__ float logisticFunction(float* x, float* w, int n, float w0) {
  float sum = w0 + dotProduct(x, w, n);
  return 1 / (1 + exp(sum));
}


void updateWeights(float* weights, float** x, float* y, float w0, float* difference) {

  for (int i = 0; i < SAMPLE_NUMBER; i++) {
    difference[i] = logisticFunction(x[i], weights, SAMPLE_ATTRIBUTE_NUMBER, w0) + y[i] - 1;
  }

  for (int j = 0; j < SAMPLE_ATTRIBUTE_NUMBER; j++) {
    for (int i = 0; i < SAMPLE_NUMBER; i++) {
      weights[j] += x[i][j] * difference[i] * CONVERGE_RATE;
    }
  }
}


__global__ void calculate_difference(float* delta, float* difference, float* x, float* weights, float* w0, float* y) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  difference[i] = logisticFunction(x + i * SAMPLE_ATTRIBUTE_NUMBER, weights, SAMPLE_ATTRIBUTE_NUMBER, *w0) + y[i] - 1;
  int delta_index_start = i * SAMPLE_ATTRIBUTE_NUMBER;
  for (int j = 0; j < SAMPLE_ATTRIBUTE_NUMBER; j++) {
    *(delta + delta_index_start + j) = *(x + delta_index_start + j) * difference[i] * CONVERGE_RATE;
  }
}
__global__ void copy_weight(float* weight_device, float* new_weights) {
  for (int i = 0; i < SAMPLE_ATTRIBUTE_NUMBER; i++) {
    weight_device[i] = new_weights[i];
  }
}

struct offset_pointers {
    __device__ float* operator()(int offset, float* delta) {
      return offset + delta;
    }
};

struct sum_delta {
    __device__ float* operator() (float* x, float* y) {
      for (int i = 0; i < SAMPLE_ATTRIBUTE_NUMBER; i++) {
        *(x + i) += *(y + i);
      }
      return x;
    }
};

void output_device_vector(float* x, int length) {
  thrust::device_ptr<float> x_thr(x);
  thrust::device_vector<float> x_vector(x_thr, x_thr + length);
  thrust::copy(x_vector.begin(), x_vector.end(), std::ostream_iterator<float>(std::cout, "\t"));
  printf("\n");
}


int main() {

  srand(time(NULL));
  // initialize the weights randomly
  float w0 = (INITIAL_WEIGHTS_RANGE * rand() / RAND_MAX) - INITIAL_WEIGHTS_RANGE / 2;
  float* weights = generateRandomVectorFloat(SAMPLE_ATTRIBUTE_NUMBER, INITIAL_WEIGHTS_RANGE);
  // TODO: load real data into x and y;
  // Generate random data for x

  float* x = (float*)malloc(SAMPLE_NUMBER * SAMPLE_ATTRIBUTE_NUMBER * sizeof(float));
  x = generateRandomVectorFloat(SAMPLE_NUMBER * SAMPLE_ATTRIBUTE_NUMBER, SAMPLE_VALUE_RANGE);

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
    y[i] = logisticFunction(x + i * SAMPLE_ATTRIBUTE_NUMBER, benchMarkWeights, SAMPLE_ATTRIBUTE_NUMBER, benchMarkWeight0) > 0.5 ? 0 : 1;
  }
  struct timeval tv;
  gettimeofday(&tv, NULL);
  long start = tv.tv_usec + tv.tv_sec * MICROSEC_IN_SEC;
  //clock_t start = clock(), diff;
  float *difference, *weight_device, *x_device, *y_device, *w0_device, *delta_device;// = (float *) malloc(sizeof(float) * SAMPLE_NUMBER);
  printf("Start memory alloc\n");
  cudaMalloc((void**)&difference, SAMPLE_NUMBER * sizeof(float));
  cudaMalloc((void**)&weight_device, SAMPLE_ATTRIBUTE_NUMBER * sizeof(float));
  cudaMalloc((void**)&delta_device, SAMPLE_ATTRIBUTE_NUMBER * SAMPLE_NUMBER * sizeof(float));
  cudaMalloc((void**)&x_device, SAMPLE_ATTRIBUTE_NUMBER * SAMPLE_NUMBER * sizeof(float));
  cudaMalloc((void**)&y_device, SAMPLE_NUMBER * sizeof(float));
  cudaMalloc((void**)&w0_device, sizeof(float));
  printf("Start memory copy\n");
  cudaMemcpy(x_device, x, SAMPLE_ATTRIBUTE_NUMBER * SAMPLE_NUMBER * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(w0_device, &w0, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(y_device, y, SAMPLE_NUMBER * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(weight_device, weights, SAMPLE_ATTRIBUTE_NUMBER * sizeof(float), cudaMemcpyHostToDevice);
  thrust::device_vector<int> offset(SAMPLE_NUMBER);
  thrust::sequence(offset.begin(), offset.end(), 0, SAMPLE_ATTRIBUTE_NUMBER);
  //thrust::copy(offset.begin(), offset.end(), std::ostream_iterator<int>(std::cout, "\n"));
  thrust::device_vector<float*> delta_pointer(SAMPLE_NUMBER);
  thrust::fill(delta_pointer.begin(), delta_pointer.end(), delta_device);
  thrust::device_vector<float*> delta_pointers(SAMPLE_NUMBER);
  thrust::transform(offset.begin(), offset.end(), delta_pointer.begin(), delta_pointers.begin(), offset_pointers());
#ifdef DEBUG
  printf("Original weights:\n");
  output_device_vector(weight_device, SAMPLE_ATTRIBUTE_NUMBER);
#endif
  int block_number = SAMPLE_NUMBER / 64;
  if (block_number == 0) {
    block_number = 1;
  }
  int thread_number = SAMPLE_NUMBER;
  if (thread_number > 64) {
    thread_number = 64;
  }
  for (int k = 0; k < ITERATION_NUMBER; k++) {
    calculate_difference<<<block_number,thread_number>>>(delta_device, difference, x_device, weight_device, w0_device, y_device);
    cudaDeviceSynchronize();
#ifdef DEBUG
    printf("x:\n");
    output_device_vector(x, SAMPLE_ATTRIBUTE_NUMBER * SAMPLE_NUMBER);
    printf("Delta:\n");
    output_device_vector(delta_device, SAMPLE_ATTRIBUTE_NUMBER * SAMPLE_NUMBER);
    printf("Difference:\n");
    output_device_vector(difference, SAMPLE_NUMBER);
#endif
    float* newWeight = thrust::reduce(delta_pointers.begin(), delta_pointers.end(), weight_device, sum_delta());
    cudaDeviceSynchronize();
#ifdef DEBUG
    printf("New weights:\n");
    output_device_vector(newWeight, SAMPLE_ATTRIBUTE_NUMBER);
#endif
    copy_weight<<<1, SAMPLE_ATTRIBUTE_NUMBER>>>(weight_device, newWeight);
    cudaDeviceSynchronize();
#ifdef DEBUG
    printf("weight_device after update:\n");
    output_device_vector(weight_device, SAMPLE_ATTRIBUTE_NUMBER);
#endif
  }
  cudaMemcpy(weights, weight_device, SAMPLE_ATTRIBUTE_NUMBER * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(x_device);
  cudaFree(y_device);
  cudaFree(weight_device);
  cudaFree(difference);
  cudaFree(w0_device);
#ifdef DEBUG
  for (int i = 0; i < SAMPLE_ATTRIBUTE_NUMBER; i++) {
    printf("Benchmark weight: %lf Estimated weight:%lf\n", benchMarkWeights[i], weights[i]);
  }
#endif
  // Predict the labels with weights estimated with logistic regression.
  float error = 0;
  for (int i = 0; i < SAMPLE_NUMBER; i++) {
    float predict = logisticFunction(x + i * SAMPLE_ATTRIBUTE_NUMBER, weights, SAMPLE_ATTRIBUTE_NUMBER, w0) > 0.5 ? 0 : 1;
#ifdef DEBUG
    printf("y[%d]: %lf Predicted: %lf\n", i, y[i], predict);
#endif
    error += fabs(predict - y[i]);
  }
  printf("Average error:%f\n", error / SAMPLE_NUMBER);
  gettimeofday(&tv, NULL);
  long diff = (tv.tv_sec * MICROSEC_IN_SEC + tv.tv_usec - start) / 1000;
  printf("Time taken: %ld seconds %ld milliseconds\n", diff / 1000, diff % 1000);
  free(x);
  free(y);
  free(weights);
  free(benchMarkWeights);
  return 0;
}



