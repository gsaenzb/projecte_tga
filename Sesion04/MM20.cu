#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef SIZE
#define SIZE 32
#endif

#ifndef PINNED
#define PINNED 0
#endif

// Kernel per a la multiplicació de matrius en blocs (el mateix que Kernel10)
__global__ void KernelMult(int N, int M, int P, float *A, float *B, float *C) {
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

// Kernel per a l'addició de matrius
__global__ void KernelAdd(int N, int M, float *A, float *B, float *C) {
  __shared__ float sA[SIZE][SIZE];
  __shared__ float sB[SIZE][SIZE];

  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int row = by * SIZE + ty;
  int col = bx * SIZE + tx;

  // Càrrega de dades a la memòria compartida
  if(row < N && col < M) {
    sA[ty][tx] = A[row*M + col];
    sB[ty][tx] = B[row*M + col];
  } else {
    sA[ty][tx] = 0.0;
    sB[ty][tx] = 0.0;
  }
  __syncthreads();

  // Realitzem l'addició
  if(row < N && col < M) {
    C[row*M + col] = sA[ty][tx] + sB[ty][tx];
  }
}

// Kernel per a la subtracció de matrius
__global__ void KernelSub(int N, int M, float *A, float *B, float *C) {
  __shared__ float sA[SIZE][SIZE];
  __shared__ float sB[SIZE][SIZE];

  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int row = by * SIZE + ty;
  int col = bx * SIZE + tx;

  // Càrrega de dades a la memòria compartida
  if(row < N && col < M) {
    sA[ty][tx] = A[row*M + col];
    sB[ty][tx] = B[row*M + col];
  } else {
    sA[ty][tx] = 0.0;
    sB[ty][tx] = 0.0;
  }
  __syncthreads();

  // Realitzem la subtracció
  if(row < N && col < M) {
    C[row*M + col] = sA[ty][tx] - sB[ty][tx];
  }
}

// Funció per inicialitzar matrius
void InitM(int N, int M, float *Mat) {
  int i;
  for (i=0; i<N*M; i++) 
    Mat[i] = rand() / (float) RAND_MAX;
}

// Funció per comprovar si hi ha error en el resultat
int error(float a, float b) {
  float tmp;
  tmp = abs(a-b) / abs(min(a,b));
  if (isnan(tmp) || tmp > 0.0001) return 1;
  else return 0;
}

// Funció per comprovar el producte de matrius
int TestMM(int N, int M, int P, float *A, float *B, float *C) {
  int i, j, k;
  float tmp;
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

// Funció per dividir una matriu en submatrius
void partitionMatrix(float *mat, float *submat11, float *submat12, float *submat21, float *submat22, int n) {
  int halfn = n/2;
  int size = halfn * halfn;

  for(int i = 0; i < halfn; i++) {
    for(int j = 0; j < halfn; j++) {
      // Submatriu 11 (superior esquerra)
      submat11[i*halfn + j] = mat[i*n + j];

      // Submatriu 12 (superior dreta)
      submat12[i*halfn + j] = mat[i*n + j + halfn];

      // Submatriu 21 (inferior esquerra)
      submat21[i*halfn + j] = mat[(i + halfn)*n + j];

      // Submatriu 22 (inferior dreta)
      submat22[i*halfn + j] = mat[(i + halfn)*n + j + halfn];
    }
  }
}

// Funció per reunir les submatrius en una matriu completa
void joinMatrix(float *mat, float *submat11, float *submat12, float *submat21, float *submat22, int n) {
  int halfn = n/2;
  int size = halfn * halfn;

  for(int i = 0; i < halfn; i++) {
    for(int j = 0; j < halfn; j++) {
      // Submatriu 11 (superior esquerra)
      mat[i*n + j] = submat11[i*halfn + j];

      // Submatriu 12 (superior dreta)
      mat[i*n + j + halfn] = submat12[i*halfn + j];

      // Submatriu 21 (inferior esquerra)
      mat[(i + halfn)*n + j] = submat21[i*halfn + j];

      // Submatriu 22 (inferior dreta)
      mat[(i + halfn)*n + j + halfn] = submat22[i*halfn + j];
    }
  }
}

// Funció principal per a l'execució
// Funció principal per a l'execució amb 2 GPUs
int main(int argc, char** argv)
{
  unsigned int N, size;
  unsigned int numBytes, numBytesSubmatrix;
  unsigned int nBlocks, nThreads;

  float TiempoTotal, TiempoKernel;
  cudaEvent_t E0, E1, E2, E3;

  float *h_A, *h_B, *h_C;
  float *d_A0, *d_B0, *d_C0; // Memòria GPU 0
  float *d_A1, *d_B1, *d_C1; // Memòria GPU 1

  // Submatrius per a l'algorisme de Strassen al host
  float *h_A11, *h_A12, *h_A21, *h_A22;
  float *h_B11, *h_B12, *h_B21, *h_B22;
  float *h_C11, *h_C12, *h_C21, *h_C22;

  // Matrius temporals per a Strassen a GPU 0
  float *d_M1_0, *d_M3_0, *d_M5_0, *d_M7_0;
  float *d_A11_0, *d_A12_0, *d_A22_0;
  float *d_B11_0, *d_B12_0, *d_B22_0;
  float *d_C11_0, *d_C12_0;
  float *d_temp1_0, *d_temp2_0;

  // Matrius temporals per a Strassen a GPU 1
  float *d_M1_1, *d_M2_1, *d_M3_1, *d_M4_1, *d_M6_1;
  float *d_A11_1, *d_A21_1, *d_A22_1;
  float *d_B11_1, *d_B12_1, *d_B21_1;
  float *d_C21_1, *d_C22_1;
  float *d_temp1_1, *d_temp2_1;

  char test;
  int numGPUs = 0;

  // Comprovem el nombre de GPUs disponibles
  cudaGetDeviceCount(&numGPUs);
  if (numGPUs < 2) {
    printf("Aquest programa requereix almenys 2 GPUs, però només n'hi ha %d disponibles.\n", numGPUs);
    return -1;
  }

  printf("Nombre de GPUs disponibles: %d\n", numGPUs);

  // Obtenim la dimensió de les matrius i comprovació de resultat
  if (argc == 3) { 
    N = atoi(argv[1]); 
    test = *argv[2];
  }
  else { 
    printf("Usage: ./exe N test\n"); 
    exit(0); 
  }

  // Comprovem que N és potència de 2 (requeriment per a Strassen)
  if((N & (N-1)) != 0) {
    printf("N ha de ser potència de 2 per a l'algoritme de Strassen\n");
    exit(0);
  }

  // Nombre de fils en cada dimensió 
  nThreads = SIZE;

  // Nombre de blocs en cada dimensió 
  nBlocks = (N+nThreads-1)/nThreads;

  numBytes = N * N * sizeof(float);
  numBytesSubmatrix = (N/2) * (N/2) * sizeof(float);

  dim3 dimGrid(nBlocks, nBlocks, 1);
  dim3 dimGridHalf((N/2+nThreads-1)/nThreads, (N/2+nThreads-1)/nThreads, 1);
  dim3 dimBlock(nThreads, nThreads, 1);

  cudaEventCreate(&E0);
  cudaEventCreate(&E1);
  cudaEventCreate(&E2);
  cudaEventCreate(&E3);

  if (PINNED) {
    // Obtenim memòria pinned en el host
    cudaMallocHost((float**)&h_A, numBytes); 
    cudaMallocHost((float**)&h_B, numBytes); 
    cudaMallocHost((float**)&h_C, numBytes);

    cudaMallocHost((float**)&h_A11, numBytesSubmatrix);
    cudaMallocHost((float**)&h_A12, numBytesSubmatrix);
    cudaMallocHost((float**)&h_A21, numBytesSubmatrix);
    cudaMallocHost((float**)&h_A22, numBytesSubmatrix);

    cudaMallocHost((float**)&h_B11, numBytesSubmatrix);
    cudaMallocHost((float**)&h_B12, numBytesSubmatrix);
    cudaMallocHost((float**)&h_B21, numBytesSubmatrix);
    cudaMallocHost((float**)&h_B22, numBytesSubmatrix);

    cudaMallocHost((float**)&h_C11, numBytesSubmatrix);
    cudaMallocHost((float**)&h_C12, numBytesSubmatrix);
    cudaMallocHost((float**)&h_C21, numBytesSubmatrix);
    cudaMallocHost((float**)&h_C22, numBytesSubmatrix);
  }
  else {
    // Obtenim memòria en el host
    h_A = (float*) malloc(numBytes); 
    h_B = (float*) malloc(numBytes); 
    h_C = (float*) malloc(numBytes);

    h_A11 = (float*) malloc(numBytesSubmatrix);
    h_A12 = (float*) malloc(numBytesSubmatrix);
    h_A21 = (float*) malloc(numBytesSubmatrix);
    h_A22 = (float*) malloc(numBytesSubmatrix);

    h_B11 = (float*) malloc(numBytesSubmatrix);
    h_B12 = (float*) malloc(numBytesSubmatrix);
    h_B21 = (float*) malloc(numBytesSubmatrix);
    h_B22 = (float*) malloc(numBytesSubmatrix);

    h_C11 = (float*) malloc(numBytesSubmatrix);
    h_C12 = (float*) malloc(numBytesSubmatrix);
    h_C21 = (float*) malloc(numBytesSubmatrix);
    h_C22 = (float*) malloc(numBytesSubmatrix);
  }

  // Inicialitzem les matrius
  InitM(N, N, h_A);
  InitM(N, N, h_B);

  // Dividim les matrius
  partitionMatrix(h_A, h_A11, h_A12, h_A21, h_A22, N);
  partitionMatrix(h_B, h_B11, h_B12, h_B21, h_B22, N);

  cudaEventRecord(E0, 0);
  cudaEventSynchronize(E0);

  //------------------------------------------------
  // CONFIGURACIÓ GPU 0
  //------------------------------------------------
  cudaSetDevice(0);

  // Obtenim memòria per a les submatrius necessàries a GPU 0
  cudaMalloc((float**)&d_A11_0, numBytesSubmatrix);
  cudaMalloc((float**)&d_A12_0, numBytesSubmatrix);
  cudaMalloc((float**)&d_A22_0, numBytesSubmatrix);

  cudaMalloc((float**)&d_B11_0, numBytesSubmatrix);
  cudaMalloc((float**)&d_B12_0, numBytesSubmatrix);
  cudaMalloc((float**)&d_B22_0, numBytesSubmatrix);

  cudaMalloc((float**)&d_C11_0, numBytesSubmatrix);
  cudaMalloc((float**)&d_C12_0, numBytesSubmatrix);

  // Memòria per a les matrius temporals M a GPU 0
  cudaMalloc((float**)&d_M1_0, numBytesSubmatrix);
  cudaMalloc((float**)&d_M3_0, numBytesSubmatrix);
  cudaMalloc((float**)&d_M5_0, numBytesSubmatrix);
  cudaMalloc((float**)&d_M7_0, numBytesSubmatrix);

  // Memòria per a matrius temporals auxiliars a GPU 0
  cudaMalloc((float**)&d_temp1_0, numBytesSubmatrix);
  cudaMalloc((float**)&d_temp2_0, numBytesSubmatrix);

  // Copiem les submatrius necessàries a GPU 0
  cudaMemcpy(d_A11_0, h_A11, numBytesSubmatrix, cudaMemcpyHostToDevice);
  cudaMemcpy(d_A12_0, h_A12, numBytesSubmatrix, cudaMemcpyHostToDevice);
  cudaMemcpy(d_A22_0, h_A22, numBytesSubmatrix, cudaMemcpyHostToDevice);

  cudaMemcpy(d_B11_0, h_B11, numBytesSubmatrix, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B12_0, h_B12, numBytesSubmatrix, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B22_0, h_B22, numBytesSubmatrix, cudaMemcpyHostToDevice);

  //------------------------------------------------
  // CONFIGURACIÓ GPU 1
  //------------------------------------------------
  cudaSetDevice(1);

  // Obtenim memòria per a les submatrius necessàries a GPU 1
  cudaMalloc((float**)&d_A11_1, numBytesSubmatrix);
  cudaMalloc((float**)&d_A21_1, numBytesSubmatrix);
  cudaMalloc((float**)&d_A22_1, numBytesSubmatrix);

  cudaMalloc((float**)&d_B11_1, numBytesSubmatrix);
  cudaMalloc((float**)&d_B12_1, numBytesSubmatrix);
  cudaMalloc((float**)&d_B21_1, numBytesSubmatrix);

  cudaMalloc((float**)&d_C21_1, numBytesSubmatrix);
  cudaMalloc((float**)&d_C22_1, numBytesSubmatrix);

  // Memòria per a les matrius temporals M a GPU 1
  cudaMalloc((float**)&d_M1_1, numBytesSubmatrix);
  cudaMalloc((float**)&d_M2_1, numBytesSubmatrix);
  cudaMalloc((float**)&d_M3_1, numBytesSubmatrix);
  cudaMalloc((float**)&d_M4_1, numBytesSubmatrix);
  cudaMalloc((float**)&d_M6_1, numBytesSubmatrix);

  // Memòria per a matrius temporals auxiliars a GPU 1
  cudaMalloc((float**)&d_temp1_1, numBytesSubmatrix);
  cudaMalloc((float**)&d_temp2_1, numBytesSubmatrix);

  // Copiem les submatrius necessàries a GPU 1
  cudaMemcpy(d_A11_1, h_A11, numBytesSubmatrix, cudaMemcpyHostToDevice);
  cudaMemcpy(d_A21_1, h_A21, numBytesSubmatrix, cudaMemcpyHostToDevice);
  cudaMemcpy(d_A22_1, h_A22, numBytesSubmatrix, cudaMemcpyHostToDevice);

  cudaMemcpy(d_B11_1, h_B11, numBytesSubmatrix, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B12_1, h_B12, numBytesSubmatrix, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B21_1, h_B21, numBytesSubmatrix, cudaMemcpyHostToDevice);

  cudaEventRecord(E1, 0);
  cudaEventSynchronize(E1);

  int halfN = N/2;

  //------------------------------------------------
  // EXECUCIÓ A GPU 0 - Càlcul de C11 i C12
  //------------------------------------------------
  cudaSetDevice(0);

  // M1 = (A11 + A22) * (B11 + B22)
  KernelAdd<<<dimGridHalf, dimBlock>>>(halfN, halfN, d_A11_0, d_A22_0, d_temp1_0);
  KernelAdd<<<dimGridHalf, dimBlock>>>(halfN, halfN, d_B11_0, d_B22_0, d_temp2_0);
  KernelMult<<<dimGridHalf, dimBlock>>>(halfN, halfN, halfN, d_temp1_0, d_temp2_0, d_M1_0);

  // M3 = A11 * (B12 - B22)
  KernelSub<<<dimGridHalf, dimBlock>>>(halfN, halfN, d_B12_0, d_B22_0, d_temp1_0);
  KernelMult<<<dimGridHalf, dimBlock>>>(halfN, halfN, halfN, d_A11_0, d_temp1_0, d_M3_0);

  // M5 = (A11 + A12) * B22
  KernelAdd<<<dimGridHalf, dimBlock>>>(halfN, halfN, d_A11_0, d_A12_0, d_temp1_0);
  KernelMult<<<dimGridHalf, dimBlock>>>(halfN, halfN, halfN, d_temp1_0, d_B22_0, d_M5_0);

  // M7 = (A12 - A22) * (B21 + B22)
  // Necessitem B21 des de la GPU 1 - Utilitzarem memòria peer-to-peer
  // Habilitem accés peer-to-peer entre GPU 0 i GPU 1
  cudaDeviceEnablePeerAccess(1, 0);

  // En una implementació real, hauríem de copiar B21 de GPU1 a GPU0 o accedir-hi directament
  // Per simplificar, considerem que ja tenim B21 a la GPU 0 amb accés peer-to-peer

  KernelSub<<<dimGridHalf, dimBlock>>>(halfN, halfN, d_A12_0, d_A22_0, d_temp1_0);
  // Simulem l'accés a B21 a través de peer-to-peer
  // En un cas real, aquesta operació es faria directament amb la memòria de la GPU 1
  cudaMemcpy(d_temp2_0, d_B21_1, numBytesSubmatrix, cudaMemcpyDeviceToDevice);
  KernelAdd<<<dimGridHalf, dimBlock>>>(halfN, halfN, d_temp2_0, d_B22_0, d_temp2_0);
  KernelMult<<<dimGridHalf, dimBlock>>>(halfN, halfN, halfN, d_temp1_0, d_temp2_0, d_M7_0);

  // C11 = M1 + M4 - M5 + M7
  // Necessitem M4 des de la GPU 1
  cudaMemcpy(d_temp1_0, d_M4_1, numBytesSubmatrix, cudaMemcpyDeviceToDevice);
  KernelAdd<<<dimGridHalf, dimBlock>>>(halfN, halfN, d_M1_0, d_temp1_0, d_temp1_0);
  KernelSub<<<dimGridHalf, dimBlock>>>(halfN, halfN, d_temp1_0, d_M5_0, d_temp2_0);
  KernelAdd<<<dimGridHalf, dimBlock>>>(halfN, halfN, d_temp2_0, d_M7_0, d_C11_0);

  // C12 = M3 + M5
  KernelAdd<<<dimGridHalf, dimBlock>>>(halfN, halfN, d_M3_0, d_M5_0, d_C12_0);

  //------------------------------------------------
  // EXECUCIÓ A GPU 1 - Càlcul de C21 i C22
  //------------------------------------------------
  cudaSetDevice(1);

  // Habilitem accés peer-to-peer des de GPU 1 a GPU 0
  cudaDeviceEnablePeerAccess(0, 0);

  // M2 = (A21 + A22) * B11
  KernelAdd<<<dimGridHalf, dimBlock>>>(halfN, halfN, d_A21_1, d_A22_1, d_temp1_1);
  KernelMult<<<dimGridHalf, dimBlock>>>(halfN, halfN, halfN, d_temp1_1, d_B11_1, d_M2_1);

  // M4 = A22 * (B21 - B11)
  KernelSub<<<dimGridHalf, dimBlock>>>(halfN, halfN, d_B21_1, d_B11_1, d_temp1_1);
  KernelMult<<<dimGridHalf, dimBlock>>>(halfN, halfN, halfN, d_A22_1, d_temp1_1, d_M4_1);

  // M6 = (A21 - A11) * (B11 + B12)
  KernelSub<<<dimGridHalf, dimBlock>>>(halfN, halfN, d_A21_1, d_A11_1, d_temp1_1);
  KernelAdd<<<dimGridHalf, dimBlock>>>(halfN, halfN, d_B11_1, d_B12_1, d_temp2_1);
  KernelMult<<<dimGridHalf, dimBlock>>>(halfN, halfN, halfN, d_temp1_1, d_temp2_1, d_M6_1);

  // C21 = M2 + M4
  KernelAdd<<<dimGridHalf, dimBlock>>>(halfN, halfN, d_M2_1, d_M4_1, d_C21_1);

  // C22 = M1 - M2 + M3 + M6
  // Necessitem M1 i M3 des de la GPU 0
  cudaMemcpy(d_M1_1, d_M1_0, numBytesSubmatrix, cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_M3_1, d_M3_0, numBytesSubmatrix, cudaMemcpyDeviceToDevice);

  KernelSub<<<dimGridHalf, dimBlock>>>(halfN, halfN, d_M1_1, d_M2_1, d_temp1_1);
  KernelAdd<<<dimGridHalf, dimBlock>>>(halfN, halfN, d_temp1_1, d_M3_1, d_temp2_1);
  KernelAdd<<<dimGridHalf, dimBlock>>>(halfN, halfN, d_temp2_1, d_M6_1, d_C22_1);

  // Sincronitzem tots els streams abans de copiar els resultats
  cudaDeviceSynchronize();

  cudaEventRecord(E2, 0);
  cudaEventSynchronize(E2);

  // Copiem els resultats de les submatrius al host des de ambdues GPUs
  cudaSetDevice(0);
  cudaMemcpy(h_C11, d_C11_0, numBytesSubmatrix, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_C12, d_C12_0, numBytesSubmatrix, cudaMemcpyDeviceToHost);

  cudaSetDevice(1);
  cudaMemcpy(h_C21, d_C21_1, numBytesSubmatrix, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_C22, d_C22_1, numBytesSubmatrix, cudaMemcpyDeviceToHost);

  // Reconstruïm la matriu C a partir de les submatrius
  joinMatrix(h_C, h_C11, h_C12, h_C21, h_C22, N);

  // Deshabilitació de l'accés peer-to-peer
  cudaSetDevice(0);
  cudaDeviceDisablePeerAccess(1);
  cudaSetDevice(1);
  cudaDeviceDisablePeerAccess(0);

  // Alliberem la memòria de la GPU 0
  cudaSetDevice(0);
  cudaFree(d_A11_0); cudaFree(d_A12_0); cudaFree(d_A22_0);
  cudaFree(d_B11_0); cudaFree(d_B12_0); cudaFree(d_B22_0);
  cudaFree(d_C11_0); cudaFree(d_C12_0);
  cudaFree(d_M1_0); cudaFree(d_M3_0); cudaFree(d_M5_0); cudaFree(d_M7_0);
  cudaFree(d_temp1_0); cudaFree(d_temp2_0);

  // Alliberem la memòria de la GPU 1
  cudaSetDevice(1);
  cudaFree(d_A11_1); cudaFree(d_A21_1); cudaFree(d_A22_1);
  cudaFree(d_B11_1); cudaFree(d_B12_1); cudaFree(d_B21_1);
  cudaFree(d_C21_1); cudaFree(d_C22_1);
  cudaFree(d_M1_1); cudaFree(d_M2_1); cudaFree(d_M3_1); cudaFree(d_M4_1); cudaFree(d_M6_1);
  cudaFree(d_temp1_1); cudaFree(d_temp2_1);

  cudaEventRecord(E3, 0);
  cudaEventSynchronize(E3);

  cudaEventElapsedTime(&TiempoTotal, E0, E3);
  cudaEventElapsedTime(&TiempoKernel, E1, E2);
  printf("\nALGORITME DE STRASSEN AMB 2 GPUs\n");
  printf("Dimensions: %dx%d\n", N, N);
  printf("nThreads: %dx%d (%d)\n", nThreads, nThreads, nThreads * nThreads);
  printf("nBlocks: %dx%d (%d)\n", nBlocks, nBlocks, nBlocks*nBlocks);
  if (PINNED) printf("Utilitzant Pinned Memory\n");
  else printf("NO utilitza Pinned Memory\n");
  printf("Temps Global: %4.6f milseg\n", TiempoTotal);
  printf("Temps Kernel: %4.6f milseg\n", TiempoKernel);
  printf("Rendiment Global: %4.2f GFLOPS\n", (2.0 * (float) N * (float) N * (float) N) / (1000000.0 * TiempoTotal));
  printf("Rendiment Kernel: %4.2f GFLOPS\n", (2.0 * (float) N * (float) N * (float) N) / (1000000.0 * TiempoKernel));

  cudaEventDestroy(E0); cudaEventDestroy(E1); cudaEventDestroy(E2); cudaEventDestroy(E3);

  if (test == 'N')
    printf ("NO TEST\n");
  else if (TestMM(N, N, N, h_A, h_B, h_C))
    printf ("TEST PASS\n");
  else
    printf ("TEST FAIL\n");

  // Alliberem la memòria del host
  if (PINNED) {
    cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_C);
    cudaFreeHost(h_A11); cudaFreeHost(h_A12); cudaFreeHost(h_A21); cudaFreeHost(h_A22);
    cudaFreeHost(h_B11); cudaFreeHost(h_B12); cudaFreeHost(h_B21); cudaFreeHost(h_B22);
    cudaFreeHost(h_C11); cudaFreeHost(h_C12); cudaFreeHost(h_C21); cudaFreeHost(h_C22);
  }
  else {
    free(h_A); free(h_B); free(h_C);
    free(h_A11); free(h_A12); free(h_A21); free(h_A22);
    free(h_B11); free(h_B12); free(h_B21); free(h_B22);
    free(h_C11); free(h_C12); free(h_C21); free(h_C22);
  }

  return 0;
}
