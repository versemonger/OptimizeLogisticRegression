CC=gcc
MPICC=mpicc
CFLAGS=-std=c11 -mtune=generic -march=native -shared-libgcc -fno-tree-vectorize -Wall
AVXFlAG=-mavx2
SSEFLAG=-msse 
OMPFLAG=-fopenmp 
CLINKS=-lm
RM=/bin/rm -f
all: naive sse avx2 omp omp-debug mpi
naive: LogisticRegressionNaive.c
	$(CC) $(CFLAGS) -o $@ LogisticRegressionNaive.c $(CLINKS)
sse: LogisticRegressionSSE.c
	$(CC) $(CFLAGS) $(SSEFLAG) -O3 -o $@ LogisticRegressionSSE.c $(CLINKS)
avx2: LogisticRegressionAVX2.c
	$(CC) $(CFLAGS) $(AVXFlAG) -O3 -o $@ LogisticRegressionAVX2.c $(CLINKS)
omp: LogisticRegressionOpenMP.c
	$(CC) $(CFLAGS) $(AVXFLAG) $(OMPFLAG) -O3 -o $@ LogisticRegressionOpenMP.c $(CLINKS)
omp-debug: LogisticRegressionOpenMP.c
	$(CC) $(CFLAGS) $(AVXFLAG) $(OMPFLAG) -O3 -g -o $@ LogisticRegressionOpenMP.c $(CLINKS)
mpi: LogisticRegressionMPI.c
	$(MPICC) $(CFLAGS) $(AVXFLAG) $(OMPFLAG) -O3 -o $@ LogisticRegressionMPI.c $(CLINKS)
clean:
	$(RM) naive sse avx2 omp omp-debug mpi
