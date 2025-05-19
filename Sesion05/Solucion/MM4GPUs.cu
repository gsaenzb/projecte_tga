#include <stdio.h>
#include <stdlib.h>

#ifndef SIZE
#define SIZE 32
#endif

// Kernel Matriz por Matriz
// C(NxM) <- A(NxP) * B (PxM)

__global__ void KernelMM(int N, int M, int P, float *A, float *B, float *C) {
//__global__ void KernelSxS (int N, int M, int P, float *A, float *B, float *C) {

  __shared__ float sA[SIZE][SIZE];
  __shared__ float sB[SIZE][SIZE];

  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int row = by * SIZE + ty;
  int col = bx * SIZE + tx;

  float tmp = 0.0;
  for (int m=0; m < P; m=m+SIZE) {
    sA[ty][tx] = A[row*P + m + tx];
    sB[ty][tx] = B[col + (m + ty)*M];
    __syncthreads();
    for (int k=0; k<SIZE; k++)
      tmp += sA[ty][k] * sB[k][tx];
    __syncthreads();
  }
  C[row*M+col] = tmp;
}


// Matriz por Matriz
// C(NxM) <- A(NxP) * B (PxM)
// Usaremos siempre N, M, P multiplos de SIZE

__global__ void Kernel1x1 (int N, int M, int P, float *A, float *B, float *C) {
//__global__ void KernelMM(int N, int M, int P, float *A, float *B, float *C) {

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float tmp = 0.0;
  for (int k=0; k<P; k++)
    tmp += A[row*P+k] * B[k*M+col];

  //if (row < N && col < M) 
    C[row*M+col] = tmp;
}


void InitM(int N, int M, float *Mat);
int TestMM(int N, int M, int P, float *A, float *B, float *C);

int nTest = 0;

// Invocacion:
// ./ejecutable TAM test
// TAM es el la dimension de las matrices
// test == 'Y', comprueba que el resultado sea correcto
// test == 'N', NO comprueba que el resultado (Util para tomar tiempos)
// Por defecto, N = 1024, test == 'N'

int main(int argc, char** argv)
{
  unsigned int N;
  unsigned int numBytesC, numBytesA, numBytesB;
  unsigned int nBlocks, nThreads;
 
  float TiempoTotal;
  cudaEvent_t E0, E1, E2, E3;
  cudaEvent_t X1, X2, X3;

  float *hA0, *hA1, *hB0, *hB1, *hC00, *hC01, *hC10, *hC11;
  float *dA0a, *dA1a, *dB0a, *dB1a, *dC00, *dC01, *dC10, *dC11;
  float *dA0b, *dA1b, *dB0b, *dB1b;

  int count;

  char test;

  // Dimension de las matrices NxN y comprobacion resultado
  if (argc == 1)      { test = 'N'; N = 1024; }
  else if (argc == 2) { test = 'N'; N = atoi(argv[1]); }
  else if (argc == 3) { test = *argv[2]; N = atoi(argv[1]); }
  else { printf("Usage: ./exe TAM test\n"); exit(0); }

  // numero de Threads en cada dimension 
  nThreads = SIZE;

  // numero de Blocks en cada dimension 
  nBlocks = (N/2)/nThreads; 
  
  numBytesC = N * N * sizeof(float) / 4;
  numBytesA = N * N * sizeof(float) / 2;
  numBytesB = N * N * sizeof(float) / 2;

  dim3 dimGrid(nBlocks, nBlocks, 1);
  dim3 dimBlock(nThreads, nThreads, 1);


  cudaGetDeviceCount(&count);

  if (count < 4) { printf("No hay suficientes GPUs\n"); exit(0); }

  // Obtiene Memoria [pinned] en el host
  // Usamos pinned Memory porque todas las comunicaciones CPU<->GPU son 
  cudaMallocHost((float**)&hA0,  numBytesA); 
  cudaMallocHost((float**)&hA1,  numBytesA); 
  cudaMallocHost((float**)&hB0,  numBytesB); 
  cudaMallocHost((float**)&hB1,  numBytesB); 
  cudaMallocHost((float**)&hC00, numBytesC); 
  cudaMallocHost((float**)&hC01, numBytesC); 
  cudaMallocHost((float**)&hC10, numBytesC); 
  cudaMallocHost((float**)&hC11, numBytesC); 

  // Inicializa las matrices
  InitM(N/2, N, hA0);
  InitM(N/2, N, hA1);
  InitM(N, N/2, hB0);
  InitM(N, N/2, hB1);


  // Obtener Memoria en cada device
  cudaSetDevice(0);
  cudaMalloc((float**)&dA0a, numBytesA); 
  cudaMalloc((float**)&dB0a, numBytesB); 
  cudaMalloc((float**)&dC00, numBytesC); 

  cudaSetDevice(1);
  cudaMalloc((float**)&dA0b, numBytesA); 
  cudaMalloc((float**)&dB1a, numBytesB); 
  cudaMalloc((float**)&dC01, numBytesC); 
  cudaEventCreate(&X1); 

  cudaSetDevice(2);
  cudaMalloc((float**)&dA1a, numBytesA); 
  cudaMalloc((float**)&dB0b, numBytesB); 
  cudaMalloc((float**)&dC10, numBytesC); 
  cudaEventCreate(&X2);

  cudaSetDevice(3);
  cudaMalloc((float**)&dA1b, numBytesA); 
  cudaMalloc((float**)&dB1b, numBytesB); 
  cudaMalloc((float**)&dC11, numBytesC); 
  cudaEventCreate(&X3);

  cudaSetDevice(0);
  cudaEventCreate(&E0);
  cudaEventCreate(&E1);
  cudaEventCreate(&E2);
  cudaEventCreate(&E3);

  cudaEventRecord(E0, 0);

  // Copiar datos desde el host en el device 
  cudaMemcpyAsync(dA0a, hA0, numBytesA, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(dB0a, hB0, numBytesB, cudaMemcpyHostToDevice);
  // Ejecutar el kernel 
  KernelMM<<<dimGrid, dimBlock>>>(N/2, N/2, N, dA0a, dB0a, dC00);

  cudaSetDevice(1);
  // Copiar datos desde el host en el device 
  cudaMemcpyAsync(dA0b, hA0, numBytesA, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(dB1a, hB1, numBytesB, cudaMemcpyHostToDevice);
  // Ejecutar el kernel 
  KernelMM<<<dimGrid, dimBlock>>>(N/2, N/2, N, dA0b, dB1a, dC01);

  cudaSetDevice(2);
  // Copiar datos desde el host en el device 
  cudaMemcpyAsync(dA1a, hA1, numBytesA, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(dB0b, hB0, numBytesB, cudaMemcpyHostToDevice);
  // Ejecutar el kernel 
  KernelMM<<<dimGrid, dimBlock>>>(N/2, N/2, N, dA1a, dB0b, dC10);

  cudaSetDevice(3);
  // Copiar datos desde el host en el device 
  cudaMemcpyAsync(dA1b, hA1, numBytesA, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(dB1b, hB1, numBytesB, cudaMemcpyHostToDevice);
  // Ejecutar el kernel 
  KernelMM<<<dimGrid, dimBlock>>>(N/2, N/2, N, dA1b, dB1b, dC11);


  cudaSetDevice(0);
  // Obtener el resultado desde el host 
  cudaMemcpyAsync(hC00, dC00, numBytesC, cudaMemcpyDeviceToHost); 

  cudaSetDevice(1);
  // Obtener el resultado desde el host 
  cudaMemcpyAsync(hC01, dC01, numBytesC, cudaMemcpyDeviceToHost); 
  cudaEventRecord(X1, 0);

  cudaSetDevice(2);
  // Obtener el resultado desde el host 
  cudaMemcpyAsync(hC10, dC10, numBytesC, cudaMemcpyDeviceToHost); 
  cudaEventRecord(X2, 0);

  cudaSetDevice(3);
  // Obtener el resultado desde el host 
  cudaMemcpyAsync(hC11, dC11, numBytesC, cudaMemcpyDeviceToHost); 
  cudaEventRecord(X3, 0);


  cudaSetDevice(0);
  cudaEventSynchronize(X1);
  cudaEventSynchronize(X2);
  cudaEventSynchronize(X3);

  cudaEventRecord(E3, 0); cudaEventSynchronize(E3);

  // Liberar Memoria del device 
  cudaSetDevice(0); cudaFree(dA0a); cudaFree(dB0a); cudaFree(dC00); 
  cudaSetDevice(1); cudaFree(dA0b); cudaFree(dB1a); cudaFree(dC01); 
  cudaSetDevice(2); cudaFree(dA1a); cudaFree(dB0b); cudaFree(dC10); 
  cudaSetDevice(3); cudaFree(dA1b); cudaFree(dB1b); cudaFree(dC11); 

  cudaEventElapsedTime(&TiempoTotal,  E0, E3);
  printf("\nKERNEL MultiGPU - Producto Matrices\n");
  printf("Dimensiones: %dx%d\n", N, N);
  printf("nThreads: %dx%d (%d)\n", nThreads, nThreads, nThreads * nThreads);
  printf("nBlocks: %dx%d (%d)\n", nBlocks, nBlocks, nBlocks*nBlocks);
  printf("Usando Pinned Memory.\n");
  printf("Tiempo Global: %4.6f milseg\n", TiempoTotal);
  printf("Rendimiento Global: %4.2f GFLOPS\n", (2.0 * (float) N * (float) N * (float) N) / (1000000.0 * TiempoTotal));
  printf("Para obtener el tiempo del kernel usad nvprof.\n");

  cudaSetDevice(0); cudaEventDestroy(E0); cudaEventDestroy(E1); cudaEventDestroy(E2); cudaEventDestroy(E3);
  cudaSetDevice(1); cudaEventDestroy(X1);
  cudaSetDevice(2); cudaEventDestroy(X2);
  cudaSetDevice(3); cudaEventDestroy(X3);

  if (test == 'N')
    printf ("NO TEST\n");
  else  if (TestMM(N/2, N/2, N, hA0, hB0, hC00) && 
            TestMM(N/2, N/2, N, hA0, hB1, hC01) && 
            TestMM(N/2, N/2, N, hA1, hB0, hC10) &&
            TestMM(N/2, N/2, N, hA1, hB1, hC11)) 
    printf ("TEST PASS\n");
  else
    printf ("TEST FAIL\n");

  cudaFreeHost(hA0); cudaFreeHost(hA1); 
  cudaFreeHost(hB0); cudaFreeHost(hB1); 
  cudaFreeHost(hC00); cudaFreeHost(hC01); cudaFreeHost(hC10); cudaFreeHost(hC11);

}


void InitM(int N, int M, float *Mat) {
   int i;
   for (i=0; i<N*M; i++) 
     Mat[i] = rand() / (float) RAND_MAX;
   
}

int error(float a, float b) {
  float tmp;

  tmp = abs(a-b) / abs(min(a,b));

  if (isnan(tmp) || tmp > 0.0001) return 1;
  else  return 0;

}

int TestMM(int N, int M, int P, float *A, float *B, float *C) {
   int i, j, k;
   float tmp;
   printf("Pass %d\n", nTest); nTest++;
   for (i=0; i<N; i++)
     for (j=0; j<M; j++) {
       tmp = 0.0;
       for (k=0; k<P; k++) 
         tmp = tmp + A[i*P+k] * B[k*M+j]; 
       if (error(tmp, C[i*M+j])) {
         printf ("%d:%d: %f - %f = %f \n", i, j, tmp, C[i*M+j], abs(tmp - C[i*M+j]));
         return 0;
       }
     }
   
   return 1;
}

