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

// STRASSEN

// Càlcul matrius M

// M1 = (A1,1 + A2,2) * (B1,1 + B2,2)
__global__ void KernelM1(int N, float *A11, float *A22, float *B11, float *B22, float *M1) {
  __shared__ float sA[SIZE][SIZE];
  __shared__ float sB[SIZE][SIZE];

  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int row = by * SIZE + ty;
  int col = bx * SIZE + tx;

  float tmp = 0.0;
  for (int m=0; m < N; m=m+SIZE) {
    // Carreguem A11+A22 i B11+B22 a la memòria compartida
    sA[ty][tx] = A11[row*N + m + tx] + A22[row*N + m + tx];
    sB[ty][tx] = B11[(m + ty)*N + col] + B22[(m + ty)*N + col];
    __syncthreads();

    for (int k=0; k<SIZE; k++)
      tmp += sA[ty][k] * sB[k][tx];
    __syncthreads();
  }
  M1[row*N+col] = tmp;
}

// M2 = (A2,1 + A2,2) * B1,1
__global__ void KernelM2(int N, float *A21, float *A22, float *B11, float *M2) {
  __shared__ float sA[SIZE][SIZE];
  __shared__ float sB[SIZE][SIZE];

  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int row = by * SIZE + ty;
  int col = bx * SIZE + tx;

  float tmp = 0.0;
  for (int m=0; m < N; m=m+SIZE) {
    // Carreguem A21+A22 i B11 a la memòria compartida
    sA[ty][tx] = A21[row*N + m + tx] + A22[row*N + m + tx];
    sB[ty][tx] = B11[(m + ty)*N + col];
    __syncthreads();

    for (int k=0; k<SIZE; k++)
      tmp += sA[ty][k] * sB[k][tx];
    __syncthreads();
  }
  M2[row*N+col] = tmp;
}

// M3 = A1,1 * (B1,2 − B2,2)
__global__ void KernelM3(int N, float *A11, float *B12, float *B22, float *M3) {
  __shared__ float sA[SIZE][SIZE];
  __shared__ float sB[SIZE][SIZE];

  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int row = by * SIZE + ty;
  int col = bx * SIZE + tx;

  float tmp = 0.0;
  for (int m=0; m < N; m=m+SIZE) {
    // Carreguem A11 i B12-B22 a la memòria compartida
    sA[ty][tx] = A11[row*N + m + tx];
    sB[ty][tx] = B12[(m + ty)*N + col] - B22[(m + ty)*N + col];
    __syncthreads();

    for (int k=0; k<SIZE; k++)
      tmp += sA[ty][k] * sB[k][tx];
    __syncthreads();
  }
  M3[row*N+col] = tmp;
}

// M4 = A2,2 * (B2,1 − B1,1)
__global__ void KernelM4(int N, float *A22, float *B21, float *B11, float *M4) {
  __shared__ float sA[SIZE][SIZE];
  __shared__ float sB[SIZE][SIZE];

  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int row = by * SIZE + ty;
  int col = bx * SIZE + tx;

  float tmp = 0.0;
  for (int m=0; m < N; m=m+SIZE) {
    // Carreguem A22 i B21-B11 a la memòria compartida
    sA[ty][tx] = A22[row*N + m + tx];
    sB[ty][tx] = B21[(m + ty)*N + col] - B11[(m + ty)*N + col];
    __syncthreads();

    for (int k=0; k<SIZE; k++)
      tmp += sA[ty][k] * sB[k][tx];
    __syncthreads();
  }
  M4[row*N+col] = tmp;
}

// M5 = (A1,1 + A1,2) * B2,2
__global__ void KernelM5(int N, float *A11, float *A12, float *B22, float *M5) {
  __shared__ float sA[SIZE][SIZE];
  __shared__ float sB[SIZE][SIZE];

  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int row = by * SIZE + ty;
  int col = bx * SIZE + tx;

  float tmp = 0.0;
  for (int m=0; m < N; m=m+SIZE) {
    // Carreguem A11+A12 i B22 a la memòria compartida
    sA[ty][tx] = A11[row*N + m + tx] + A12[row*N + m + tx];
    sB[ty][tx] = B22[(m + ty)*N + col];
    __syncthreads();

    for (int k=0; k<SIZE; k++)
      tmp += sA[ty][k] * sB[k][tx];
    __syncthreads();
  }
  M5[row*N+col] = tmp;
}

// M6 = (A2,1 − A1,1) * (B1,1 + B1,2)
__global__ void KernelM6(int N, float *A21, float *A11, float *B11, float *B12, float *M6) {
  __shared__ float sA[SIZE][SIZE];
  __shared__ float sB[SIZE][SIZE];

  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int row = by * SIZE + ty;
  int col = bx * SIZE + tx;

  float tmp = 0.0;
  for (int m=0; m < N; m=m+SIZE) {
    // Carreguem A21-A11 i B11+B12 a la memòria compartida
    sA[ty][tx] = A21[row*N + m + tx] - A11[row*N + m + tx];
    sB[ty][tx] = B11[(m + ty)*N + col] + B12[(m + ty)*N + col];
    __syncthreads();

    for (int k=0; k<SIZE; k++)
      tmp += sA[ty][k] * sB[k][tx];
    __syncthreads();
  }
  M6[row*N+col] = tmp;
}

// M7 = (A1,2 − A2,2) * (B2,1 + B2,2)
__global__ void KernelM7(int N, float *A12, float *A22, float *B21, float *B22, float *M7) {
  __shared__ float sA[SIZE][SIZE];
  __shared__ float sB[SIZE][SIZE];

  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int row = by * SIZE + ty;
  int col = bx * SIZE + tx;

  float tmp = 0.0;
  for (int m=0; m < N; m=m+SIZE) {
    // Carreguem A12-A22 i B21+B22 a la memòria compartida
    sA[ty][tx] = A12[row*N + m + tx] - A22[row*N + m + tx];
    sB[ty][tx] = B21[(m + ty)*N + col] + B22[(m + ty)*N + col];
    __syncthreads();

    for (int k=0; k<SIZE; k++)
      tmp += sA[ty][k] * sB[k][tx];
    __syncthreads();
  }
  M7[row*N+col] = tmp;
}

// Càlcul matrius C

// C1,1 = M1 + M4 − M5 + M7
__global__ void KernelC11(int N, float *M1, float *M4, float *M5, float *M7, float *C11) {
  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int row = by * SIZE + ty;
  int col = bx * SIZE + tx;

  if (row < N && col < N) {
    int idx = row * N + col;
    C11[idx] = M1[idx] + M4[idx] - M5[idx] + M7[idx];
  }
}

// C1,2 = M3 + M5
__global__ void KernelC12(int N, float *M3, float *M5, float *C12) {
  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int row = by * SIZE + ty;
  int col = bx * SIZE + tx;

  if (row < N && col < N) {
    int idx = row * N + col;
    C12[idx] = M3[idx] + M5[idx];
  }
}

// C2,1 = M2 + M4
__global__ void KernelC21(int N, float *M2, float *M4, float *C21) {
  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int row = by * SIZE + ty;
  int col = bx * SIZE + tx;

  if (row < N && col < N) {
    int idx = row * N + col;
    C21[idx] = M2[idx] + M4[idx];
  }
}

// C2,2 = M1 − M2 + M3 + M6
__global__ void KernelC22(int N, float *M1, float *M2, float *M3, float *M6, float *C22) {
  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int row = by * SIZE + ty;
  int col = bx * SIZE + tx;

  if (row < N && col < N) {
    int idx = row * N + col;
    C22[idx] = M1[idx] - M2[idx] + M3[idx] + M6[idx];
  }
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

int main(int argc, char** argv) {
  unsigned int N;
  unsigned int numBytes, numBytesHalf;
  unsigned int nBlocks, nThreads;

  float TiempoTotal, TiempoKernel;
  cudaEvent_t E0, E1, E2, E3;
  cudaEvent_t X1, X2, X3;

  // Matrius host
  float *hA, *hB, *hC, *hCCheck;

  // Matrius i submatrius de GPU 0 (calcularà C11)
  float *dA0, *dB0;
  float *dA0_11, *dA0_22;
  float *dB0_11, *dB0_21, *dB0_22;
  float *dC0_11;
  float *dM0_1, *dM0_4, *dM0_5, *dM0_7;

  // Matrius i submatrius de GPU 1 (calcularà C12)
  float *dA1, *dB1;
  float *dA1_11, *dA1_12;
  float *dB1_12, *dB1_22;
  float *dC1_12;
  float *dM1_3, *dM1_5;

  // Matrius i submatrius de GPU 2 (calcularà C21)
  float *dA2, *dB2;
  float *dA2_21, *dA2_22;
  float *dB2_11, *dB2_21;
  float *dC2_21;
  float *dM2_2, *dM2_4;

  // Matrius i submatrius de GPU 3 (calcularà C22)
  float *dA3, *dB3;
  float *dA3_11, *dA3_21, *dA3_22;
  float *dB3_11, *dB3_12, *dB3_22;
  float *dC3_22;
  float *dM3_1, *dM3_2, *dM3_3, *dM3_6;

  int count;
  char test;

  // Dimension de les matrius NxN i comprovació del resultat
  if (argc == 1)      { test = 'N'; N = 1024; }
  else if (argc == 2) { test = 'N'; N = atoi(argv[1]); }
  else if (argc == 3) { test = *argv[2]; N = atoi(argv[1]); }
  else { printf("Usage: ./exe TAM test\n"); exit(0); }

  // Comprovem que la dimensió sigui potència de 2
  if ((N & (N-1)) != 0) {
    printf("La dimensió ha de ser potència de 2\n");
    exit(0);
  }

  // número de Threads en cada dimensió 
  nThreads = SIZE;

  // número de Blocks en cada dimensió (per a mida N/2)
  nBlocks = (N/2)/nThreads;

  numBytes = N * N * sizeof(float);
  numBytesHalf = (N/2) * (N/2) * sizeof(float);

  dim3 dimGrid(nBlocks, nBlocks, 1);
  dim3 dimBlock(nThreads, nThreads, 1);

  cudaGetDeviceCount(&count);
  if (count < 4) { printf("No hay suficientes GPUs\n"); exit(0); }

  // Obtenim memòria al host
  cudaMallocHost((float**)&hA, numBytes);
  cudaMallocHost((float**)&hB, numBytes);
  cudaMallocHost((float**)&hC, numBytes);
  cudaMallocHost((float**)&hCCheck, numBytes);

  // Inicialitzem les matrius
  InitM(N, N, hA);
  InitM(N, N, hB);

  // Definim mida de mitja matriu
  int half_N = N/2;

  // Configuració GPU 0 (C11)
  cudaSetDevice(0);
  cudaMalloc((float**)&dA0, numBytes);
  cudaMalloc((float**)&dB0, numBytes);

  // Només les submatrius que necessitem per C11
  cudaMalloc((float**)&dA0_11, numBytesHalf);
  cudaMalloc((float**)&dA0_22, numBytesHalf);
  cudaMalloc((float**)&dB0_11, numBytesHalf);
  cudaMalloc((float**)&dB0_21, numBytesHalf);
  cudaMalloc((float**)&dB0_22, numBytesHalf);

  cudaMalloc((float**)&dC0_11, numBytesHalf);

  cudaMalloc((float**)&dM0_1, numBytesHalf);
  cudaMalloc((float**)&dM0_4, numBytesHalf);
  cudaMalloc((float**)&dM0_5, numBytesHalf);
  cudaMalloc((float**)&dM0_7, numBytesHalf);

  // Configuració GPU 1 (C12)
  cudaSetDevice(1);
  cudaMalloc((float**)&dA1, numBytes);
  cudaMalloc((float**)&dB1, numBytes);

  // Només les submatrius que necessitem per C12
  cudaMalloc((float**)&dA1_11, numBytesHalf);
  cudaMalloc((float**)&dA1_12, numBytesHalf);
  cudaMalloc((float**)&dB1_12, numBytesHalf);
  cudaMalloc((float**)&dB1_22, numBytesHalf);

  cudaMalloc((float**)&dC1_12, numBytesHalf);

  cudaMalloc((float**)&dM1_3, numBytesHalf);
  cudaMalloc((float**)&dM1_5, numBytesHalf);

  cudaEventCreate(&X1);

  // Configuració GPU 2 (C21)
  cudaSetDevice(2);
  cudaMalloc((float**)&dA2, numBytes);
  cudaMalloc((float**)&dB2, numBytes);

  // Només les submatrius que necessitem per C21
  cudaMalloc((float**)&dA2_21, numBytesHalf);
  cudaMalloc((float**)&dA2_22, numBytesHalf);
  cudaMalloc((float**)&dB2_11, numBytesHalf);
  cudaMalloc((float**)&dB2_21, numBytesHalf);

  cudaMalloc((float**)&dC2_21, numBytesHalf);

  cudaMalloc((float**)&dM2_2, numBytesHalf);
  cudaMalloc((float**)&dM2_4, numBytesHalf);

  cudaEventCreate(&X2);

  // Configuració GPU 3 (C22)
  cudaSetDevice(3);
  cudaMalloc((float**)&dA3, numBytes);
  cudaMalloc((float**)&dB3, numBytes);

  // Només les submatrius que necessitem per C22
  cudaMalloc((float**)&dA3_11, numBytesHalf);
  cudaMalloc((float**)&dA3_21, numBytesHalf);
  cudaMalloc((float**)&dA3_22, numBytesHalf);
  cudaMalloc((float**)&dB3_11, numBytesHalf);
  cudaMalloc((float**)&dB3_12, numBytesHalf);
  cudaMalloc((float**)&dB3_22, numBytesHalf);

  cudaMalloc((float**)&dC3_22, numBytesHalf);

  cudaMalloc((float**)&dM3_1, numBytesHalf);
  cudaMalloc((float**)&dM3_2, numBytesHalf);
  cudaMalloc((float**)&dM3_3, numBytesHalf);
  cudaMalloc((float**)&dM3_6, numBytesHalf);

  cudaEventCreate(&X3);

  // Tornem a la GPU 0 per començar els càlculs
  cudaSetDevice(0);
  cudaEventCreate(&E0);
  cudaEventCreate(&E1);
  cudaEventCreate(&E2);
  cudaEventCreate(&E3);

  // Copiem les matrius A i B al device per a tots els GPUs
  cudaSetDevice(0);
  cudaMemcpy(dA0, hA, numBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(dB0, hB, numBytes, cudaMemcpyHostToDevice);

  cudaSetDevice(1);
  cudaMemcpy(dA1, hA, numBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(dB1, hB, numBytes, cudaMemcpyHostToDevice);

  cudaSetDevice(2);
  cudaMemcpy(dA2, hA, numBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(dB2, hB, numBytes, cudaMemcpyHostToDevice);

  cudaSetDevice(3);
  cudaMemcpy(dA3, hA, numBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(dB3, hB, numBytes, cudaMemcpyHostToDevice);

  cudaSetDevice(0);
  cudaEventRecord(E0, 0);

  // Iniciem el cronòmetre per mesurar el temps dels kernels
  cudaEventRecord(E1, 0);

  // GPU 0: Extreure les submatrius A i B necessàries per C11
  cudaSetDevice(0);
  for (int i = 0; i < half_N; i++) {
    cudaMemcpy(&dA0_11[i*half_N], &hA[i*N], half_N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&dA0_22[i*half_N], &hA[(i+half_N)*N+half_N], half_N*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(&dB0_11[i*half_N], &hB[i*N], half_N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&dB0_21[i*half_N], &hB[(i+half_N)*N], half_N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&dB0_22[i*half_N], &hB[(i+half_N)*N+half_N], half_N*sizeof(float), cudaMemcpyHostToDevice);
  }

  // GPU 1: Extreure les submatrius A i B necessàries per C12
  cudaSetDevice(1);
  for (int i = 0; i < half_N; i++) {
    cudaMemcpy(&dA1_11[i*half_N], &hA[i*N], half_N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&dA1_12[i*half_N], &hA[i*N+half_N], half_N*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(&dB1_12[i*half_N], &hB[i*N+half_N], half_N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&dB1_22[i*half_N], &hB[(i+half_N)*N+half_N], half_N*sizeof(float), cudaMemcpyHostToDevice);
  }

  // GPU 2: Extreure les submatrius A i B necessàries per C21
  cudaSetDevice(2);
  for (int i = 0; i < half_N; i++) {
    cudaMemcpy(&dA2_21[i*half_N], &hA[(i+half_N)*N], half_N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&dA2_22[i*half_N], &hA[(i+half_N)*N+half_N], half_N*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(&dB2_11[i*half_N], &hB[i*N], half_N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&dB2_21[i*half_N], &hB[(i+half_N)*N], half_N*sizeof(float), cudaMemcpyHostToDevice);
  }

  // GPU 3: Extreure les submatrius A i B necessàries per C22
  cudaSetDevice(3);
  for (int i = 0; i < half_N; i++) {
    cudaMemcpy(&dA3_11[i*half_N], &hA[i*N], half_N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&dA3_21[i*half_N], &hA[(i+half_N)*N], half_N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&dA3_22[i*half_N], &hA[(i+half_N)*N+half_N], half_N*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(&dB3_11[i*half_N], &hB[i*N], half_N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&dB3_12[i*half_N], &hB[i*N+half_N], half_N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&dB3_22[i*half_N], &hB[(i+half_N)*N+half_N], half_N*sizeof(float), cudaMemcpyHostToDevice);
  }
  // GPU 0: Calculem les matrius M necessàries per C11
  cudaSetDevice(0);
  // M1 = (A1,1 + A2,2) · (B1,1 + B2,2)
  KernelM1<<<dimGrid, dimBlock>>>(half_N, dA0_11, dA0_22, dB0_11, dB0_22, dM0_1);
  // M4 = A2,2 · (B2,1 − B1,1)
  KernelM4<<<dimGrid, dimBlock>>>(half_N, dA0_22, dB0_21, dB0_11, dM0_4);
  // Per a M5, necessitem A1,2, però no el tenim a la GPU 0 -->> Decisió Gian 1: Copiar A1,2 de la host memory a la GPU0
  float *dA0_12;
  cudaMalloc((float**)&dA0_12, numBytesHalf);
  for (int i = 0; i < half_N; i++) {
    cudaMemcpy(&dA0_12[i*half_N], &hA[i*N+half_N], half_N*sizeof(float), cudaMemcpyHostToDevice);
  }
  // M5 = (A1,1 + A1,2) · B2,2
  KernelM5<<<dimGrid, dimBlock>>>(half_N, dA0_11, dA0_12, dB0_22, dM0_5);
  cudaFree(dA0_12);

  // M7 = (A1,2 − A2,2) · (B2,1 + B2,2)
  cudaMalloc((float**)&dA0_12, numBytesHalf);
  for (int i = 0; i < half_N; i++) {
    cudaMemcpy(&dA0_12[i*half_N], &hA[i*N+half_N], half_N*sizeof(float), cudaMemcpyHostToDevice);
  }
  KernelM7<<<dimGrid, dimBlock>>>(half_N, dA0_12, dA0_22, dB0_21, dB0_22, dM0_7);
  cudaFree(dA0_12);

  // GPU 1: Calculem les matrius M necessàries per C12
  cudaSetDevice(1);
  // M3 = A1,1 · (B1,2 − B2,2)
  KernelM3<<<dimGrid, dimBlock>>>(half_N, dA1_11, dB1_12, dB1_22, dM1_3);
  // M5 = (A1,1 + A1,2) · B2,2
  KernelM5<<<dimGrid, dimBlock>>>(half_N, dA1_11, dA1_12, dB1_22, dM1_5);

  // GPU 2: Calculem les matrius M necessàries per C21
  cudaSetDevice(2);
  // M2 = (A2,1 + A2,2) · B1,1
  KernelM2<<<dimGrid, dimBlock>>>(half_N, dA2_21, dA2_22, dB2_11, dM2_2);
  // M4 = A2,2 · (B2,1 − B1,1)
  KernelM4<<<dimGrid, dimBlock>>>(half_N, dA2_22, dB2_21, dB2_11, dM2_4);

  // GPU 3: Calculem les matrius M necessàries per C22
  cudaSetDevice(3);
  // M1 = (A1,1 + A2,2) · (B1,1 + B2,2)
  KernelM1<<<dimGrid, dimBlock>>>(half_N, dA3_11, dA3_22, dB3_11, dB3_22, dM3_1);
  // M2 = (A2,1 + A2,2) · B1,1
  KernelM2<<<dimGrid, dimBlock>>>(half_N, dA3_21, dA3_22, dB3_11, dM3_2);
  // M3 = A1,1 · (B1,2 − B2,2)
  KernelM3<<<dimGrid, dimBlock>>>(half_N, dA3_11, dB3_12, dB3_22, dM3_3);
  // M6 = (A2,1 − A1,1) · (B1,1 + B1,2)
  KernelM6<<<dimGrid, dimBlock>>>(half_N, dA3_21, dA3_11, dB3_11, dB3_12, dM3_6);

  // Calculem les matrius C

  // GPU 0: Calculem C11
  cudaSetDevice(0);
  // C1,1 = M1 + M4 − M5 + M7
  KernelC11<<<dimGrid, dimBlock>>>(half_N, dM0_1, dM0_4, dM0_5, dM0_7, dC0_11);

  // GPU 1: Calculem C12
  cudaSetDevice(1);
  // C1,2 = M3 + M5
  KernelC12<<<dimGrid, dimBlock>>>(half_N, dM1_3, dM1_5, dC1_12);

  // GPU 2: Calculem C21
  cudaSetDevice(2);
  // C2,1 = M2 + M4
  KernelC21<<<dimGrid, dimBlock>>>(half_N, dM2_2, dM2_4, dC2_21);

  // GPU 3: Calculem C22
  cudaSetDevice(3);
  // C2,2 = M1 − M2 + M3 + M6
  KernelC22<<<dimGrid, dimBlock>>>(half_N, dM3_1, dM3_2, dM3_3, dM3_6, dC3_22);

  // Finalitzem el cronòmetre dels kernels
  cudaSetDevice(0);
  cudaEventRecord(E2, 0);

  // Unir les submatrius C per obtenir el resultat final
  // Copiem les dades cap al host
  cudaSetDevice(0);
  for (int i = 0; i < half_N; i++) {
    cudaMemcpy(&hC[i*N], &dC0_11[i*half_N], half_N*sizeof(float), cudaMemcpyDeviceToHost);
  }

  cudaSetDevice(1);
  for (int i = 0; i < half_N; i++) {
    cudaMemcpy(&hC[i*N+half_N], &dC1_12[i*half_N], half_N*sizeof(float), cudaMemcpyDeviceToHost);
  }
  cudaEventRecord(X1, 0);

  cudaSetDevice(2);
  for (int i = 0; i < half_N; i++) {
    cudaMemcpy(&hC[(i+half_N)*N], &dC2_21[i*half_N], half_N*sizeof(float), cudaMemcpyDeviceToHost);
  }
  cudaEventRecord(X2, 0);

  cudaSetDevice(3);
  for (int i = 0; i < half_N; i++) {
    cudaMemcpy(&hC[(i+half_N)*N+half_N], &dC3_22[i*half_N], half_N*sizeof(float), cudaMemcpyDeviceToHost);
  }
  cudaEventRecord(X3, 0);

  cudaSetDevice(0);
  cudaEventSynchronize(X1);
  cudaEventSynchronize(X2);
  cudaEventSynchronize(X3);
  cudaEventRecord(E3, 0);
  cudaEventSynchronize(E3);

  // Alliberem memòria al device
  cudaSetDevice(0);
  cudaFree(dA0); cudaFree(dB0);
  cudaFree(dA0_11); cudaFree(dA0_22);
  cudaFree(dB0_11); cudaFree(dB0_21); cudaFree(dB0_22);
  cudaFree(dC0_11);
  cudaFree(dM0_1); cudaFree(dM0_4); cudaFree(dM0_5); cudaFree(dM0_7);

  cudaSetDevice(1);
  cudaFree(dA1); cudaFree(dB1);
  cudaFree(dA1_11); cudaFree(dA1_12);
  cudaFree(dB1_12); cudaFree(dB1_22);
  cudaFree(dC1_12);
  cudaFree(dM1_3); cudaFree(dM1_5);

  cudaSetDevice(2);
  cudaFree(dA2); cudaFree(dB2);
  cudaFree(dA2_21); cudaFree(dA2_22);
  cudaFree(dB2_11); cudaFree(dB2_21);
  cudaFree(dC2_21);
  cudaFree(dM2_2); cudaFree(dM2_4);

  cudaSetDevice(3);
  cudaFree(dA3); cudaFree(dB3);
  cudaFree(dA3_11); cudaFree(dA3_21); cudaFree(dA3_22);
  cudaFree(dB3_11); cudaFree(dB3_12); cudaFree(dB3_22);
  cudaFree(dC3_22);
  cudaFree(dM3_1); cudaFree(dM3_2); cudaFree(dM3_3); cudaFree(dM3_6);

  cudaSetDevice(0);
  cudaEventElapsedTime(&TiempoTotal, E0, E3);
  cudaEventElapsedTime(&TiempoKernel, E1, E2);

  printf("\nKERNEL Strassen 4 GPUs - Producto Matrices\n");
  printf("Dimensiones: %dx%d\n", N, N);
  printf("nThreads: %dx%d (%d)\n", nThreads, nThreads, nThreads * nThreads);
  printf("nBlocks: %dx%d (%d)\n", nBlocks, nBlocks, nBlocks*nBlocks);
  printf("Tiempo Global: %4.6f milseg\n", TiempoTotal);
  printf("Tiempo Kernel: %4.6f milseg\n", TiempoKernel);
  printf("Rendimiento Global: %4.2f GFLOPS\n", (2.0 * (float) N * (float) N * (float) N) / (1000000.0 * TiempoTotal));
  printf("Rendimiento Kernel: %4.2f GFLOPS\n", (2.0 * (float) N * (float) N * (float) N) / (1000000.0 * TiempoKernel));

  cudaSetDevice(0); cudaEventDestroy(E0); cudaEventDestroy(E1); cudaEventDestroy(E2); cudaEventDestroy(E3);
  cudaSetDevice(1); cudaEventDestroy(X1);
  cudaSetDevice(2); cudaEventDestroy(X2);
  cudaSetDevice(3); cudaEventDestroy(X3);

  if (test == 'N')
    printf("NO TEST\n");
  else {
    // Calculem el producte a la CPU per verificar
    float *cpuC = (float*)malloc(numBytes);

    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        cpuC[i*N+j] = 0.0;
        for (int k = 0; k < N; k++)
          cpuC[i*N+j] += hA[i*N+k] * hB[k*N+j];
      }
    }

    // Comparem els resultats
    bool correct = true;
    for (int i = 0; i < N*N; i++) {
      if (fabs(cpuC[i] - hC[i]) > 0.0001) {
        correct = false;
        printf("Error a la posició %d: CPU=%f, GPU=%f\n", i, cpuC[i], hC[i]);
        break;
      }
    }

    if (correct)
      printf("TEST PASS\n");
    else
      printf("TEST FAIL\n");

    free(cpuC);
  }

  cudaFreeHost(hA); cudaFreeHost(hB); cudaFreeHost(hC); cudaFreeHost(hCCheck);

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

