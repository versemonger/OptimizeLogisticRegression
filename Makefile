CC=gcc
CFLAGS=-std=c11 -mtune=generic -march=native -shared-libgcc -fno-tree-vectorize -Wall
AVXFlAG=-mavx2 -O3
SSEFLAG=-msse -O3
OMPFLAG=-fopenmp -O3
CLINKS=-lm
RM=/bin/rm -f
all: naive sse avx2 omp omp-debug
naive: LogisticRegressionNaive.c
	$(CC) $(CFLAGS) -o $@ LogisticRegressionNaive.c $(CLINKS)
sse: LogisticRegressionSSE.c
	$(CC) $(CFLAGS) $(SSEFLAG) -o $@ LogisticRegressionSSE.c $(CLINKS)
avx2: LogisticRegressionAVX2.c
	$(CC) $(CFLAGS) $(AVXFlAG) -o $@ LogisticRegressionAVX2.c $(CLINKS)
omp: LogisticRegressionOpenMP.c
	$(CC) $(CFLAGS) $(AVXFLAG) $(OMPFLAG) -o $@ LogisticRegressionOpenMP.c $(CLINKS)
omp-debug: LogisticRegressionOpenMP.c
	$(CC) $(CFLAGS) $(AVXFLAG) $(OMPFLAG) -g -o $@ LogisticRegressionOpenMP.c $(CLINKS)
clean:
	$(RM) naive sse avx2 omp omp-debug
