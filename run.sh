#All changes to code are copyright, 2017, Zhu Li, zhuli@unm.edu

echo "Naive version"
gcc LogisticRegressionNaive.c -std=c11 -lm -o naive -mtune=generic -march=native -shared-libgcc -fno-tree-vectorize -O3 -Wall
./naive

echo "SSE version"
gcc LogisticRegressionSSE.c -std=c11 -lm -o sse -mtune=generic -march=native -shared-libgcc -fno-tree-vectorize -O3 -Wall -msse
./sse

echo "AVX2 version"
gcc LogisticRegressionAVX2.c -std=c11 -lm -o avx2 -mtune=generic -march=native -shared-libgcc -fno-tree-vectorize -O3 -Wall -mavx2
./avx2
