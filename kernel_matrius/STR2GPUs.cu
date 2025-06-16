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
    sA[ty][tx] = A11[by*N*SIZE + m + ty*N + tx] + A22[by*N*SIZE + m + ty*N + tx];
    sB[ty][tx] = B11[m*N + bx*SIZE + ty*N + tx] + B22[m*N + bx*SIZE + ty*N + tx];
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
    sA[ty][tx] = A21[by*N*SIZE + m + ty*N + tx] + A22[by*N*SIZE + m + ty*N + tx];
    sB[ty][tx] = B11[m*N + bx*SIZE + ty*N + tx];
    __syncthreads();

    for (int k=0; k<SIZE; k++)
      tmp += sA[ty][k] * sB[k][tx];
    __syncthreads();
  }
  M2[row*N+col] = tmp;
}

// M3 = A1,1 * (B1,2 - B2,2)
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
    sA[ty][tx] = A11[by*N*SIZE + m + ty*N + tx];
    sB[ty][tx] = B12[m*N + bx*SIZE + ty*N + tx] - B22[m*N + bx*SIZE + ty*N + tx];
    __syncthreads();

    for (int k=0; k<SIZE; k++)
      tmp += sA[ty][k] * sB[k][tx];
    __syncthreads();
  }
  M3[row*N+col] = tmp;
}

// M4 = A2,2 * (B2,1 - B1,1)
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
    sA[ty][tx] = A22[by*N*SIZE + m + ty*N + tx];
    sB[ty][tx] = B21[m*N + bx*SIZE + ty*N + tx] - B11[m*N + bx*SIZE + ty*N + tx];
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
    sA[ty][tx] = A11[by*N*SIZE + m + ty*N + tx] + A12[by*N*SIZE + m + ty*N + tx];
    sB[ty][tx] = B22[m*N + bx*SIZE + ty*N + tx];
    __syncthreads();

    for (int k=0; k<SIZE; k++)
      tmp += sA[ty][k] * sB[k][tx];
    __syncthreads();
  }
  M5[row*N+col] = tmp;
}

// M6 = (A2,1 + A1,1) * (B1,1 + B1,2)
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
    sA[ty][tx] = A21[by*N*SIZE + m + ty*N + tx] - A11[by*N*SIZE + m + ty*N + tx];
    sB[ty][tx] = B11[m*N + bx*SIZE + ty*N + tx] + B12[m*N + bx*SIZE + ty*N + tx];
    __syncthreads();

    for (int k=0; k<SIZE; k++)
      tmp += sA[ty][k] * sB[k][tx];
    __syncthreads();
  }
  M6[row*N+col] = tmp;
}

// M7 = (A1,2 - A2,2) * (B2,1 + B2,2)
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
    sA[ty][tx] = A12[by*N*SIZE + m + ty*N + tx] - A22[by*N*SIZE + m + ty*N + tx];
    sB[ty][tx] = B21[m*N + bx*SIZE + ty*N + tx] + B22[m*N + bx*SIZE + ty*N + tx];
    __syncthreads();

    for (int k=0; k<SIZE; k++)
      tmp += sA[ty][k] * sB[k][tx];
    __syncthreads();
  }
  M7[row*N+col] = tmp;
}

// Càlcul matrius C

// C1,1 = M1 + M4 + M5 + M7
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

// C2,2 = M1 - M2 + M3 + M6
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

// Funció per dividir una matriu en submatriu
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


void sumMatrixCPU(float *A, float *B, float *C, int nr) {
  for (int i = 0; i < nr; i++)
    for (int j = 0; j < nr; j++)
      C[i*nr + j] = A[i*nr + j] + B[i*nr + j]; 
}
void subMatrixCPU(float *A, float *B, float *C, int nr) {
  for (int i = 0; i < nr; i++)
    for (int j = 0; j < nr; j++)
      C[i*nr + j] = A[i*nr + j] - B[i*nr + j]; 
}
void mulMatrixCPU(float *A, float *B, float *C, int nr) {
  for (int i = 0; i < nr; i++) {
    for (int j = 0; j < nr; j++) {
      C[i*nr+j] = 0.0;
      for (int k = 0; k < nr; k++)
        C[i*nr+j] += A[i*nr+k] * B[k*nr+j];
    }
  }
}

void computeStrassenCPU(float *A11, float *A12, float *A21, float *A22,
                        float *B11, float *B12, float *B21, float *B22,
                        float *M1,  float *M2,  float *M3,  float *M4,  float *M5, float *M6, float *M7, 
                        float *C11, float *C12, float *C21, float *C22, float *C, int N) {
  int half_N = N/2;
  int numBytesHalf = half_N * half_N * sizeof(float);

  float *tmp1 = (float *)malloc(numBytesHalf);
  float *tmp2 = (float *)malloc(numBytesHalf);

  // M1 = (A11 + A22) * (B11 + B22)
  sumMatrixCPU(A11, A22, tmp1, half_N);
  sumMatrixCPU(B11, B22, tmp2, half_N);
  mulMatrixCPU(tmp1, tmp2, M1, half_N);
  // M2 = (A21 + A22) * B11
  sumMatrixCPU(A21, A22, tmp1, half_N);
  mulMatrixCPU(tmp1, B11, M2, half_N);
  // M3 = A11 * (B12 - B22)
  subMatrixCPU(B12, B22, tmp2, half_N);
  mulMatrixCPU(A11, tmp2, M3, half_N);
  // M4 = A22 * (B21 - B11)
  subMatrixCPU(B21, B11, tmp2, half_N);
  mulMatrixCPU(A22, tmp2, M4, half_N);
  // M5 = (A11 + A12) * B22 
  sumMatrixCPU(A11, A12, tmp1, half_N);
  mulMatrixCPU(tmp1, B22, M5, half_N);
  // M6 = (A21 - A11) * (B11 + B12)
  subMatrixCPU(A21, A11, tmp1, half_N);
  sumMatrixCPU(B11, B12, tmp2, half_N);
  mulMatrixCPU(tmp1, tmp2, M6, half_N);
  // M7 = (A12 - A22) * (B21 + B22)
  subMatrixCPU(A12, A22, tmp1, half_N);
  sumMatrixCPU(B21, B22, tmp2, half_N);
  mulMatrixCPU(tmp1, tmp2, M7, half_N);
  // C11 = M1 + M4 - M5 + M7
  sumMatrixCPU(M1, M4, tmp1, half_N);
  subMatrixCPU(tmp1, M5, tmp2, half_N);
  sumMatrixCPU(tmp2, M7, C11, half_N);
  // C12 = M3 + M5
  sumMatrixCPU(M3, M5, C12, half_N);
  // C21 = M2 + M4
  sumMatrixCPU(M2, M4, C21, half_N);
  // C22 = M1 - M2 + M3 + M6
  subMatrixCPU(M1, M2, tmp1, half_N);
  sumMatrixCPU(tmp1, M3, tmp2, half_N);
  sumMatrixCPU(tmp2, M6, C22, half_N);
  joinMatrix(C, C11, C12, C21, C22, N);

  free(tmp1);
  free(tmp2);
}


// Invocacion:
// ./ejecutable TAM test
// TAM es el la dimension de las matrices
// test == 'Y', comprueba que el resultado sea correcto
// test == 'N', NO comprueba que el resultado (Util para tomar tiempos)
// Por defecto, N = 1024, test == 'N'

int main(int argc, char** argv) {
  unsigned int N, half_N;
  unsigned int numBytes, numBytesHalf;
  unsigned int nBlocks, nThreads;

  float TiempoTotal, TiempoKernel;
  cudaEvent_t E0, E1, E2, E3, E4;

  float *hA, *hB, *hC;
  float *hA11, *hA12, *hA21, *hA22;
  float *hB11, *hB12, *hB21, *hB22;
  float *hC11, *hC12, *hC21, *hC22;
  // Pointers to each of the devices
  float *d0A21, *d0A11, *d0A12, *d0A22, *d0B11, *d0B12, *d0B21, *d0B22, *d0M1 , *d0M3, *d0M4, *d0M5, *d0M7;
  float *d1A21, *d1A11, *d1A12, *d1A22, *d1B11, *d1B12, *d1B21, *d1B22, *d1M1 , *d1M2 , *d1M3, *d1M4 , *d1M6;
  float *d0C11, *d0C12, *d1C21, *d1C22;

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
  
  int count;
  cudaGetDeviceCount(&count);
  if (count < 2) { printf("No hi ha suficients GPUs\n"); exit(0); }
  
  half_N = N/2;

  // número de Threads en cada dimensió 
  nThreads = SIZE;

  // número de Blocks en cada dimensió (per a mida N/2)
  nBlocks = (half_N)/nThreads;

  numBytes = N * N * sizeof(float);
  numBytesHalf = (half_N) * (half_N) * sizeof(float);

  dim3 dimGrid(nBlocks, nBlocks, 1);
  dim3 dimBlock(nThreads, nThreads, 1);

  // Obtenim memòria al host
  cudaMallocHost((float**)&hA, numBytes);
  cudaMallocHost((float**)&hB, numBytes);
  cudaMallocHost((float**)&hC, numBytes);

  cudaMallocHost((float**)&hA11, numBytesHalf);
  cudaMallocHost((float**)&hA12, numBytesHalf);
  cudaMallocHost((float**)&hA21, numBytesHalf);
  cudaMallocHost((float**)&hA22, numBytesHalf);
  cudaMallocHost((float**)&hB11, numBytesHalf);
  cudaMallocHost((float**)&hB12, numBytesHalf);
  cudaMallocHost((float**)&hB21, numBytesHalf);
  cudaMallocHost((float**)&hB22, numBytesHalf);
  cudaMallocHost((float**)&hC11, numBytesHalf);
  cudaMallocHost((float**)&hC12, numBytesHalf);
  cudaMallocHost((float**)&hC21, numBytesHalf);
  cudaMallocHost((float**)&hC22, numBytesHalf);
  
  // Inicialitzem les matrius al host
  InitM(N, N, hA);
  InitM(N, N, hB);
  
  // Particionem les matriu
  partitionMatrix(hA, hA11, hA12, hA21, hA22, N);
  partitionMatrix(hB, hB11, hB12, hB21, hB22, N);

  // Reserve memory for each of the devices
  cudaSetDevice(0);
  cudaMalloc((float**)&d0A11, numBytesHalf);
  cudaMalloc((float**)&d0A12, numBytesHalf);
  cudaMalloc((float**)&d0A22, numBytesHalf);
  cudaMalloc((float**)&d0A21, numBytesHalf);
  cudaMalloc((float**)&d0B11, numBytesHalf);
  cudaMalloc((float**)&d0B12, numBytesHalf);
  cudaMalloc((float**)&d0B21, numBytesHalf);
  cudaMalloc((float**)&d0B22, numBytesHalf);
  cudaMalloc((float**)&d0M1 , numBytesHalf);
  cudaMalloc((float**)&d0M3 , numBytesHalf);
  cudaMalloc((float**)&d0M4 , numBytesHalf);
  cudaMalloc((float**)&d0M5 , numBytesHalf);
  cudaMalloc((float**)&d0M7 , numBytesHalf);
  cudaMalloc((float**)&d0C11 , numBytesHalf);
  cudaMalloc((float**)&d0C12 , numBytesHalf);
  
  cudaSetDevice(1);
  cudaMalloc((float**)&d1A11, numBytesHalf);
  cudaMalloc((float**)&d1A12, numBytesHalf);
  cudaMalloc((float**)&d1A21, numBytesHalf);
  cudaMalloc((float**)&d1A22, numBytesHalf);
  cudaMalloc((float**)&d1B11, numBytesHalf);
  cudaMalloc((float**)&d1B12, numBytesHalf);
  cudaMalloc((float**)&d1B21, numBytesHalf);
  cudaMalloc((float**)&d1B22, numBytesHalf);
  cudaMalloc((float**)&d1M1 , numBytesHalf);
  cudaMalloc((float**)&d1M2 , numBytesHalf);
  cudaMalloc((float**)&d1M3 , numBytesHalf);
  cudaMalloc((float**)&d1M4 , numBytesHalf);
  cudaMalloc((float**)&d1M6 , numBytesHalf);
  cudaMalloc((float**)&d1C21 , numBytesHalf);
  cudaMalloc((float**)&d1C22 , numBytesHalf);
  
  cudaSetDevice(0);
  cudaEventCreate(&E0);
  cudaEventCreate(&E1);
  cudaEventCreate(&E2);
  cudaEventCreate(&E4);

  cudaSetDevice(1); cudaEventCreate(&E3);

  // Implementem l'algorisme de Strassen
  
  cudaSetDevice(0);
  cudaEventRecord(E0, 0);
  
  cudaMemcpy(d0A11, hA11, numBytesHalf, cudaMemcpyHostToDevice);
  cudaMemcpy(d0A12, hA12, numBytesHalf, cudaMemcpyHostToDevice);
  cudaMemcpy(d0A21, hA21, numBytesHalf, cudaMemcpyHostToDevice);
  cudaMemcpy(d0A22, hA22, numBytesHalf, cudaMemcpyHostToDevice);
  cudaMemcpy(d0B11, hB11, numBytesHalf, cudaMemcpyHostToDevice);
  cudaMemcpy(d0B12, hB12, numBytesHalf, cudaMemcpyHostToDevice);
  cudaMemcpy(d0B21, hB21, numBytesHalf, cudaMemcpyHostToDevice);
  cudaMemcpy(d0B22, hB22, numBytesHalf, cudaMemcpyHostToDevice);
  
  cudaEventRecord(E1, 0);

  cudaSetDevice(1);
  cudaMemcpy(d1A11, hA11, numBytesHalf, cudaMemcpyHostToDevice);
  cudaMemcpy(d1A12, hA12, numBytesHalf, cudaMemcpyHostToDevice);
  cudaMemcpy(d1A21, hA21, numBytesHalf, cudaMemcpyHostToDevice);
  cudaMemcpy(d1A22, hA22, numBytesHalf, cudaMemcpyHostToDevice);
  cudaMemcpy(d1B11, hB11, numBytesHalf, cudaMemcpyHostToDevice);
  cudaMemcpy(d1B12, hB12, numBytesHalf, cudaMemcpyHostToDevice);
  cudaMemcpy(d1B21, hB21, numBytesHalf, cudaMemcpyHostToDevice);
  cudaMemcpy(d1B22, hB22, numBytesHalf, cudaMemcpyHostToDevice);

  cudaSetDevice(0);

  // GPU 0: Calculem les matrius M necessàries per C11 i C12
  // cudaSetDevice(0);
  KernelM1<<<dimGrid, dimBlock>>>(half_N, d0A11, d0A22, d0B11, d0B22, d0M1);
  KernelM3<<<dimGrid, dimBlock>>>(half_N, d0A11, d0B12, d0B22, d0M3);
  KernelM5<<<dimGrid, dimBlock>>>(half_N, d0A11, d0A12, d0B22, d0M5);
  KernelM4<<<dimGrid, dimBlock>>>(half_N, d0A22, d0B21, d0B11, d0M4);
  KernelM7<<<dimGrid, dimBlock>>>(half_N, d0A12, d0A22, d0B21, d0B22, d0M7);
  // cudaMemcpy(d1M1, d0M1, numBytesHalf, cudaMemcpyDeviceToDevice);

  // GPU 1: Calculem les matrius M necessàries per C21 i C22
  cudaSetDevice(1);
  KernelM1<<<dimGrid, dimBlock>>>(half_N, d1A11, d1A22, d1B11, d1B22, d1M1);
  KernelM2<<<dimGrid, dimBlock>>>(half_N, d1A21, d1A22, d1B11, d1M2);
  KernelM3<<<dimGrid, dimBlock>>>(half_N, d1A11, d1B12, d1B22, d1M3);
  KernelM4<<<dimGrid, dimBlock>>>(half_N, d1A22, d1B21, d1B11, d1M4);
  KernelM6<<<dimGrid, dimBlock>>>(half_N, d1A21, d1A11, d1B11, d1B12, d1M6);
  // cudaMemcpy(d0M1, d1M1, numBytesHalf, cudaMemcpyDeviceToDevice);

  // 3. Calculem les submatrius C
  cudaSetDevice(0);
  KernelC11<<<dimGrid, dimBlock>>>(half_N, d0M1, d0M4, d0M5, d0M7, d0C11);
  KernelC12<<<dimGrid, dimBlock>>>(half_N, d0M3, d0M5, d0C12);

  cudaSetDevice(1);
  KernelC21<<<dimGrid, dimBlock>>>(half_N, d1M2, d1M4, d1C21);
  KernelC22<<<dimGrid, dimBlock>>>(half_N, d1M1, d1M2, d1M3, d1M6, d1C22);

  // Finalitzem el cronòmetre dels kernels
  cudaSetDevice(0);
  cudaEventRecord(E2, 0);

  // 4. Unir les submatrius C per obtenir el resultat final
  // cudaSetDevice(0);
  cudaMemcpy(hC11, d0C11, numBytesHalf, cudaMemcpyDeviceToHost);
  cudaMemcpy(hC12, d0C12, numBytesHalf, cudaMemcpyDeviceToHost);
  cudaSetDevice(1);
  cudaMemcpy(hC21, d1C21, numBytesHalf, cudaMemcpyDeviceToHost);
  cudaMemcpy(hC22, d1C22, numBytesHalf, cudaMemcpyDeviceToHost);
  cudaEventRecord(E3, 0);
  
  cudaSetDevice(0); cudaEventSynchronize(E3);
  cudaEventRecord(E4, 0); cudaEventSynchronize(E4);
  
  joinMatrix(hC, hC11, hC12, hC21, hC22, N);
  
  // Alliberem memòria al device
  

  cudaEventElapsedTime(&TiempoTotal, E0, E4);
  cudaEventElapsedTime(&TiempoKernel, E1, E2);

  printf("\nKERNEL Strassen 1 GPU - Producto Matrices\n");
  printf("Dimensiones: %dx%d\n", N, N);
  printf("nThreads: %dx%d (%d)\n", nThreads, nThreads, nThreads * nThreads);
  printf("nBlocks: %dx%d (%d)\n", nBlocks, nBlocks, nBlocks*nBlocks);
  printf("Tiempo Global: %4.6f milseg\n", TiempoTotal);
  printf("Tiempo Kernel: %4.6f milseg\n", TiempoKernel);
  printf("Rendimiento Global: %4.2f GFLOPS\n", (2.0 * (float) N * (float) N * (float) N) / (1000000.0 * TiempoTotal));
  printf("Rendimiento Kernel: %4.2f GFLOPS\n", (2.0 * (float) N * (float) N * (float) N) / (1000000.0 * TiempoKernel));

  cudaEventDestroy(E0); cudaEventDestroy(E1); cudaEventDestroy(E2); cudaEventDestroy(E3); cudaEventDestroy(E4);

  if (test == 'N')
    printf("NO TEST\n");
  else {
    // Calculem el producte a la CPU per verificar
    float *cpuC = (float*)malloc(numBytes);
    
    float *M1 = (float*)malloc(numBytesHalf);
    float *M2 = (float*)malloc(numBytesHalf);
    float *M3 = (float*)malloc(numBytesHalf);
    float *M4 = (float*)malloc(numBytesHalf);
    float *M5 = (float*)malloc(numBytesHalf);
    float *M6 = (float*)malloc(numBytesHalf);
    float *M7 = (float*)malloc(numBytesHalf);

    float *C11 = (float*)malloc(numBytesHalf);
    float *C12 = (float*)malloc(numBytesHalf);
    float *C21 = (float*)malloc(numBytesHalf);
    float *C22 = (float*)malloc(numBytesHalf);
  
    computeStrassenCPU(hA11, hA12, hA21, hA22,
                       hB11, hB12, hB21, hB22,
                       M1, M2, M3, M4, M5, M6, M7, 
                       C11, C12, C21, C22, cpuC, N);

    free(M1); free(M2); free(M3); free(M4); free(M5); free(M6); free(M7);
    free(C11); free(C12); free(C21); free(C22);

    // for (int i = 0; i < N; i++) {
    //   for (int j = 0; j < N; j++) {
    //     cpuC[i*N+j] = 0.0;
    //     for (int k = 0; k < N; k++)
    //       cpuC[i*N+j] += hA[i*N+k] * hB[k*N+j];
    //   }
    // }

    // Comparem els resultats
    bool correct = true;
    for (int i = 0; i < N*N; i++) {
      if (fabs(cpuC[i] - hC[i]) > 0.001) {
        correct = false;
        printf("Error a la posició %d: CPU=%f, GPU=%f\n", i, cpuC[i], hC[i]);
        break;
      }
    }
    
    // for (int i = 0; i < 8; i++) {
    //   printf(" %f", cpuC[i]);
    // }
    // printf("\n");
    // for (int i = 0; i < 8; i++) {
    //   printf(" %f", hC[i]);
    // }
    // printf("\n");


    if (correct)
      printf("TEST PASS\n");
    else
      printf("TEST FAIL\n");

    free(cpuC);
  }

  cudaFreeHost(hA); cudaFreeHost(hB);
}

void InitM(int N, int M, float *Mat) {
   int i;
   for (i=0; i<N*M; i++) 
     Mat[i] = rand() / (float) RAND_MAX;
   
}

int error(float a, float b) {
  float tmp;

  tmp = abs(a-b) / abs(min(a,b));

  if (isnan(tmp) || tmp > 0.001) return 1;
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

