#include <stdio.h>
#include <stdlib.h>

#ifndef SIZE
#define SIZE 32
#endif

// Kernel Matriz por Matriz
// C(NxM) <- A(NxP) * B (PxM)

__global__ void KernelMM(int N, int M, int P, float *A, float *B, float *C) {
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

// STRASSEN KERNELS per 4 GPUs

// M1 = (A11 + A22) * (B11 + B22)
__global__ void KernelM1(int N, float *A11, float *A22, float *B11, float *B22, float *M1) {
  __shared__ float sA[SIZE][SIZE];
  __shared__ float sB[SIZE][SIZE];

  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int row = by * SIZE + ty;
  int col = bx * SIZE + tx;

  float tmp = 0.0;
  for (int m=0; m < N; m=m+SIZE) {
    sA[ty][tx] = A11[by*N*SIZE + m + ty*N + tx] + A22[by*N*SIZE + m + ty*N + tx];
    sB[ty][tx] = B11[m*N + bx*SIZE + ty*N + tx] + B22[m*N + bx*SIZE + ty*N + tx];
    __syncthreads();

    for (int k=0; k<SIZE; k++)
      tmp += sA[ty][k] * sB[k][tx];
    __syncthreads();
  }
  M1[row*N+col] = tmp;
}

// M2 = (A21 + A22) * B11
__global__ void KernelM2(int N, float *A21, float *A22, float *B11, float *M2) {
  __shared__ float sA[SIZE][SIZE];
  __shared__ float sB[SIZE][SIZE];

  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int row = by * SIZE + ty;
  int col = bx * SIZE + tx;

  float tmp = 0.0;
  for (int m=0; m < N; m=m+SIZE) {
    sA[ty][tx] = A21[by*N*SIZE + m + ty*N + tx] + A22[by*N*SIZE + m + ty*N + tx];
    sB[ty][tx] = B11[m*N + bx*SIZE + ty*N + tx];
    __syncthreads();

    for (int k=0; k<SIZE; k++)
      tmp += sA[ty][k] * sB[k][tx];
    __syncthreads();
  }
  M2[row*N+col] = tmp;
}

// M3 = A11 * (B12 - B22)
__global__ void KernelM3(int N, float *A11, float *B12, float *B22, float *M3) {
  __shared__ float sA[SIZE][SIZE];
  __shared__ float sB[SIZE][SIZE];

  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int row = by * SIZE + ty;
  int col = bx * SIZE + tx;

  float tmp = 0.0;
  for (int m=0; m < N; m=m+SIZE) {
    sA[ty][tx] = A11[by*N*SIZE + m + ty*N + tx];
    sB[ty][tx] = B12[m*N + bx*SIZE + ty*N + tx] - B22[m*N + bx*SIZE + ty*N + tx];
    __syncthreads();

    for (int k=0; k<SIZE; k++)
      tmp += sA[ty][k] * sB[k][tx];
    __syncthreads();
  }
  M3[row*N+col] = tmp;
}

// M4 = A22 * (B21 - B11)
__global__ void KernelM4(int N, float *A22, float *B21, float *B11, float *M4) {
  __shared__ float sA[SIZE][SIZE];
  __shared__ float sB[SIZE][SIZE];

  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int row = by * SIZE + ty;
  int col = bx * SIZE + tx;

  float tmp = 0.0;
  for (int m=0; m < N; m=m+SIZE) {
    sA[ty][tx] = A22[by*N*SIZE + m + ty*N + tx];
    sB[ty][tx] = B21[m*N + bx*SIZE + ty*N + tx] - B11[m*N + bx*SIZE + ty*N + tx];
    __syncthreads();

    for (int k=0; k<SIZE; k++)
      tmp += sA[ty][k] * sB[k][tx];
    __syncthreads();
  }
  M4[row*N+col] = tmp;
}

// M5 = (A11 + A12) * B22
__global__ void KernelM5(int N, float *A11, float *A12, float *B22, float *M5) {
  __shared__ float sA[SIZE][SIZE];
  __shared__ float sB[SIZE][SIZE];

  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int row = by * SIZE + ty;
  int col = bx * SIZE + tx;

  float tmp = 0.0;
  for (int m=0; m < N; m=m+SIZE) {
    sA[ty][tx] = A11[by*N*SIZE + m + ty*N + tx] + A12[by*N*SIZE + m + ty*N + tx];
    sB[ty][tx] = B22[m*N + bx*SIZE + ty*N + tx];
    __syncthreads();

    for (int k=0; k<SIZE; k++)
      tmp += sA[ty][k] * sB[k][tx];
    __syncthreads();
  }
  M5[row*N+col] = tmp;
}

// M6 = (A21 - A11) * (B11 + B12)
__global__ void KernelM6(int N, float *A21, float *A11, float *B11, float *B12, float *M6) {
  __shared__ float sA[SIZE][SIZE];
  __shared__ float sB[SIZE][SIZE];

  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int row = by * SIZE + ty;
  int col = bx * SIZE + tx;

  float tmp = 0.0;
  for (int m=0; m < N; m=m+SIZE) {
    sA[ty][tx] = A21[by*N*SIZE + m + ty*N + tx] - A11[by*N*SIZE + m + ty*N + tx];
    sB[ty][tx] = B11[m*N + bx*SIZE + ty*N + tx] + B12[m*N + bx*SIZE + ty*N + tx];
    __syncthreads();

    for (int k=0; k<SIZE; k++)
      tmp += sA[ty][k] * sB[k][tx];
    __syncthreads();
  }
  M6[row*N+col] = tmp;
}

// M7 = (A12 - A22) * (B21 + B22)
__global__ void KernelM7(int N, float *A12, float *A22, float *B21, float *B22, float *M7) {
  __shared__ float sA[SIZE][SIZE];
  __shared__ float sB[SIZE][SIZE];

  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int row = by * SIZE + ty;
  int col = bx * SIZE + tx;

  float tmp = 0.0;
  for (int m=0; m < N; m=m+SIZE) {
    sA[ty][tx] = A12[by*N*SIZE + m + ty*N + tx] - A22[by*N*SIZE + m + ty*N + tx];
    sB[ty][tx] = B21[m*N + bx*SIZE + ty*N + tx] + B22[m*N + bx*SIZE + ty*N + tx];
    __syncthreads();

    for (int k=0; k<SIZE; k++)
      tmp += sA[ty][k] * sB[k][tx];
    __syncthreads();
  }
  M7[row*N+col] = tmp;
}

// Kernels per calcular les submatrius C

// C11 = M1 + M4 - M5 + M7
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

// C12 = M3 + M5
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

// C21 = M2 + M4
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

// C22 = M1 - M2 + M3 + M6
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

// Funció per dividir una matriu en 4 submatrius (Strassen)
void partitionMatrix(float *mat, float *submat11, float *submat12, float *submat21, float *submat22, int n) {
  int halfn = n/2;

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

// Funció per reunir les 4 submatrius en una matriu completa
void joinMatrix(float *mat, float *submat11, float *submat12, float *submat21, float *submat22, int n) {
  int halfn = n/2;

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

// Funció per copiar matrius GPU-GPU directament
void copyMatrixGPUtoGPU(float *src, float *dst, int size, int srcGPU, int dstGPU, cudaStream_t stream) {
  cudaMemcpyPeerAsync(dst, dstGPU, src, srcGPU, size * sizeof(float), stream);
}

// Main amb 4 GPUs - Strassen Algorithm
int main(int argc, char** argv) {
  unsigned int N, half_N;
  unsigned int numBytes, numBytesHalf;
  unsigned int nBlocks, nThreads;
  int numGPUs = 4;

  float TiempoTotal, TiempoKernel;
  cudaEvent_t E0, E1, E2, E3;

  // Host arrays
  float *hA, *hB, *hC;
  float *hA11, *hA12, *hA21, *hA22;
  float *hB11, *hB12, *hB21, *hB22;
  float *hC11, *hC12, *hC21, *hC22;

  // Device arrays per cada GPU
  float *dA11[4], *dA12[4], *dA21[4], *dA22[4];
  float *dB11[4], *dB12[4], *dB21[4], *dB22[4];
  float *dC11[4], *dC12[4], *dC21[4], *dC22[4];
  float *dM1[4], *dM2[4], *dM3[4], *dM4[4], *dM5[4], *dM6[4], *dM7[4];

  // Streams per cada GPU
  cudaStream_t stream[4];

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

  // Comprovem que tenim 4 GPUs disponibles
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount < 4) {
    printf("Necesitem almenys 4 GPUs. Només hi ha %d disponibles.\n", deviceCount);
    exit(0);
  }

  printf("Trobades %d GPUs disponibles\n", deviceCount);

  // Habilitem comunicació P2P entre totes les GPUs
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      if (i != j) {
        int canAccessPeer;
        cudaDeviceCanAccessPeer(&canAccessPeer, i, j);
        if (canAccessPeer) {
          cudaSetDevice(i);
          cudaError_t err = cudaDeviceEnablePeerAccess(j, 0);
          if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
            printf("Error habilitant P2P entre GPU %d i GPU %d: %s\n", i, j, cudaGetErrorString(err));
          }
        } else {
          printf("P2P no disponible entre GPU %d i GPU %d\n", i, j);
        }
      }
    }
  }

  printf("Starting Strassen algorithm with 4 GPUs, N=%d\n", N);

  half_N = N/2;
  nThreads = SIZE;
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

  // Particionem les matrius
  partitionMatrix(hA, hA11, hA12, hA21, hA22, N);
  partitionMatrix(hB, hB11, hB12, hB21, hB22, N);

  // Alloquem memòria a les 4 GPUs i creem streams
  for (int gpu = 0; gpu < 4; gpu++) {
    cudaSetDevice(gpu);

    // Només alloquem les matrius necessàries per cada GPU segons les taules
    cudaMalloc((float**)&dA11[gpu], numBytesHalf);
    cudaMalloc((float**)&dA12[gpu], numBytesHalf);
    cudaMalloc((float**)&dA21[gpu], numBytesHalf);
    cudaMalloc((float**)&dA22[gpu], numBytesHalf);
    cudaMalloc((float**)&dB11[gpu], numBytesHalf);
    cudaMalloc((float**)&dB12[gpu], numBytesHalf);
    cudaMalloc((float**)&dB21[gpu], numBytesHalf);
    cudaMalloc((float**)&dB22[gpu], numBytesHalf);

    // Matrius M necessàries per cada GPU
    cudaMalloc((float**)&dM1[gpu], numBytesHalf);
    cudaMalloc((float**)&dM2[gpu], numBytesHalf);
    cudaMalloc((float**)&dM3[gpu], numBytesHalf);
    cudaMalloc((float**)&dM4[gpu], numBytesHalf);
    cudaMalloc((float**)&dM5[gpu], numBytesHalf);
    cudaMalloc((float**)&dM6[gpu], numBytesHalf);
    cudaMalloc((float**)&dM7[gpu], numBytesHalf);

    // Matrius C
    cudaMalloc((float**)&dC11[gpu], numBytesHalf);
    cudaMalloc((float**)&dC12[gpu], numBytesHalf);
    cudaMalloc((float**)&dC21[gpu], numBytesHalf);
    cudaMalloc((float**)&dC22[gpu], numBytesHalf);

    cudaStreamCreate(&stream[gpu]);
  }

  // Creem events a la GPU 0
  cudaSetDevice(0);
  cudaEventCreate(&E0);
  cudaEventCreate(&E1);
  cudaEventCreate(&E2);
  cudaEventCreate(&E3);

  // Implementem l'algorisme de Strassen amb 4 GPUs
  cudaSetDevice(0);
  cudaEventRecord(E0, stream[0]);
  cudaEventSynchronize(E0);

  // Copiem totes les submatrius a totes les GPUs
  for (int gpu = 0; gpu < 4; gpu++) {
    cudaSetDevice(gpu);
    cudaMemcpyAsync(dA11[gpu], hA11, numBytesHalf, cudaMemcpyHostToDevice, stream[gpu]);
    cudaMemcpyAsync(dA12[gpu], hA12, numBytesHalf, cudaMemcpyHostToDevice, stream[gpu]);
    cudaMemcpyAsync(dA21[gpu], hA21, numBytesHalf, cudaMemcpyHostToDevice, stream[gpu]);
    cudaMemcpyAsync(dA22[gpu], hA22, numBytesHalf, cudaMemcpyHostToDevice, stream[gpu]);
    cudaMemcpyAsync(dB11[gpu], hB11, numBytesHalf, cudaMemcpyHostToDevice, stream[gpu]);
    cudaMemcpyAsync(dB12[gpu], hB12, numBytesHalf, cudaMemcpyHostToDevice, stream[gpu]);
    cudaMemcpyAsync(dB21[gpu], hB21, numBytesHalf, cudaMemcpyHostToDevice, stream[gpu]);
    cudaMemcpyAsync(dB22[gpu], hB22, numBytesHalf, cudaMemcpyHostToDevice, stream[gpu]);
  }

  // Sincronitzem totes les transferències
  for (int gpu = 0; gpu < 4; gpu++) {
    cudaSetDevice(gpu);
    cudaStreamSynchronize(stream[gpu]);
  }

  cudaSetDevice(0);
  cudaEventRecord(E1, stream[0]);
  cudaEventSynchronize(E1);

  // Estratègia per 4 GPUs: cada GPU calcula les matrius M que necessita per la seva submatriu C
  // Segons les taules de dependències:

  // GPU 0 calcula C11: necessita M1, M4, M5, M7
  cudaSetDevice(0);
  KernelM1<<<dimGrid, dimBlock, 0, stream[0]>>>(half_N, dA11[0], dA22[0], dB11[0], dB22[0], dM1[0]);
  KernelM4<<<dimGrid, dimBlock, 0, stream[0]>>>(half_N, dA22[0], dB21[0], dB11[0], dM4[0]);
  KernelM5<<<dimGrid, dimBlock, 0, stream[0]>>>(half_N, dA11[0], dA12[0], dB22[0], dM5[0]);
  KernelM7<<<dimGrid, dimBlock, 0, stream[0]>>>(half_N, dA12[0], dA22[0], dB21[0], dB22[0], dM7[0]);

  // GPU 1 calcula C12: necessita M3, M5
  cudaSetDevice(1);
  KernelM3<<<dimGrid, dimBlock, 0, stream[1]>>>(half_N, dA11[1], dB12[1], dB22[1], dM3[1]);
  KernelM5<<<dimGrid, dimBlock, 0, stream[1]>>>(half_N, dA11[1], dA12[1], dB22[1], dM5[1]);

  // GPU 2 calcula C21: necessita M2, M4
  cudaSetDevice(2);
  KernelM2<<<dimGrid, dimBlock, 0, stream[2]>>>(half_N, dA21[2], dA22[2], dB11[2], dM2[2]);
  KernelM4<<<dimGrid, dimBlock, 0, stream[2]>>>(half_N, dA22[2], dB21[2], dB11[2], dM4[2]);

  // GPU 3 calcula C22: necessita M1, M2, M3, M6
  cudaSetDevice(3);
  KernelM1<<<dimGrid, dimBlock, 0, stream[3]>>>(half_N, dA11[3], dA22[3], dB11[3], dB22[3], dM1[3]);
  KernelM2<<<dimGrid, dimBlock, 0, stream[3]>>>(half_N, dA21[3], dA22[3], dB11[3], dM2[3]);
  KernelM3<<<dimGrid, dimBlock, 0, stream[3]>>>(half_N, dA11[3], dB12[3], dB22[3], dM3[3]);
  KernelM6<<<dimGrid, dimBlock, 0, stream[3]>>>(half_N, dA21[3], dA11[3], dB11[3], dB12[3], dM6[3]);

  // Sincronitzem tots els kernels M
  for (int gpu = 0; gpu < 4; gpu++) {
    cudaSetDevice(gpu);
    cudaStreamSynchronize(stream[gpu]);
  }

  // Calculem les submatrius C a cada GPU
  cudaSetDevice(0);
  KernelC11<<<dimGrid, dimBlock, 0, stream[0]>>>(half_N, dM1[0], dM4[0], dM5[0], dM7[0], dC11[0]);

  cudaSetDevice(1);
  KernelC12<<<dimGrid, dimBlock, 0, stream[1]>>>(half_N, dM3[1], dM5[1], dC12[1]);

  cudaSetDevice(2);
  KernelC21<<<dimGrid, dimBlock, 0, stream[2]>>>(half_N, dM2[2], dM4[2], dC21[2]);

  cudaSetDevice(3);
  KernelC22<<<dimGrid, dimBlock, 0, stream[3]>>>(half_N, dM1[3], dM2[3], dM3[3], dM6[3], dC22[3]);

  // Sincronitzem tots els kernels C
  for (int gpu = 0; gpu < 4; gpu++) {
    cudaSetDevice(gpu);
    cudaStreamSynchronize(stream[gpu]);
  }

  // Finalitzem el cronòmetre dels kernels
  cudaSetDevice(0);
  cudaEventRecord(E2, stream[0]);
  cudaEventSynchronize(E2);

  // Copiem els resultats de cada GPU al host
  cudaSetDevice(0);
  cudaMemcpyAsync(hC11, dC11[0], numBytesHalf, cudaMemcpyDeviceToHost, stream[0]);

  cudaSetDevice(1);
  cudaMemcpyAsync(hC12, dC12[1], numBytesHalf, cudaMemcpyDeviceToHost, stream[1]);

  cudaSetDevice(2);
  cudaMemcpyAsync(hC21, dC21[2], numBytesHalf, cudaMemcpyDeviceToHost, stream[2]);

  cudaSetDevice(3);
  cudaMemcpyAsync(hC22, dC22[3], numBytesHalf, cudaMemcpyDeviceToHost, stream[3]);

  // Sincronitzem totes les transferències
  for (int gpu = 0; gpu < 4; gpu++) {
    cudaSetDevice(gpu);
    cudaStreamSynchronize(stream[gpu]);
  }

  cudaSetDevice(0);
  cudaEventRecord(E3, stream[0]);
  cudaEventSynchronize(E3);

  // Unir les submatrius C per obtenir el resultat final
  joinMatrix(hC, hC11, hC12, hC21, hC22, N);

  // Alliberem memòria a les 4 GPUs
  for (int gpu = 0; gpu < 4; gpu++) {
    cudaSetDevice(gpu);

    cudaFree(dA11[gpu]); cudaFree(dA12[gpu]); cudaFree(dA21[gpu]); cudaFree(dA22[gpu]);
    cudaFree(dB11[gpu]); cudaFree(dB12[gpu]); cudaFree(dB21[gpu]); cudaFree(dB22[gpu]);
    cudaFree(dC11[gpu]); cudaFree(dC12[gpu]); cudaFree(dC21[gpu]); cudaFree(dC22[gpu]);
    cudaFree(dM1[gpu]);  cudaFree(dM2[gpu]);  cudaFree(dM3[gpu]);  cudaFree(dM4[gpu]);
    cudaFree(dM5[gpu]);  cudaFree(dM6[gpu]);  cudaFree(dM7[gpu]);

    cudaStreamDestroy(stream[gpu]);
  }

  // Calculem els temps d'execució
  cudaSetDevice(0);
  cudaEventElapsedTime(&TiempoTotal, E0, E3);
  cudaEventElapsedTime(&TiempoKernel, E1, E2);

  printf("\nKERNEL Strassen 4 GPUs - Producto Matrices\n");
  printf("Dimensiones: %dx%d\n", N, N);
  printf("nThreads: %dx%d (%d)\n", nThreads, nThreads, nThreads * nThreads);
  printf("nBlocks: %dx%d (%d)\n", nBlocks, nBlocks, nBlocks*nBlocks);
  printf("Tiempo Global: %4.6f milseg\n", TiempoTotal);
  printf("Tiempo Kernel: %4.6f milseg\n", TiempoKernel);

  if (TiempoTotal > 0.0) {
    printf("Rendimiento Global: %4.2f GFLOPS\n", (2.0 * (float) N * (float) N * (float) N) / (1000000.0 * TiempoTotal));
  } else {
    printf("Rendimiento Global: No mesurable (temps massa petit)\n");
  }

  if (TiempoKernel > 0.0) {
    printf("Rendimiento Kernel: %4.2f GFLOPS\n", (2.0 * (float) N * (float) N * (float) N) / (1000000.0 * TiempoKernel));
  } else {
    printf("Rendimiento Kernel: No mesurable (temps massa petit)\n");
  }

  cudaEventDestroy(E0); cudaEventDestroy(E1); cudaEventDestroy(E2); cudaEventDestroy(E3);

  if (test == 'N')
    printf("NO TEST\n");
  else {
    // Calculem el producte amb CPU per verificar (versió simplificada de Strassen)
    printf("Verificant resultat...\n");

    // Fem una verificació parcial comparant alguns elements
    float *cpuC = (float*)malloc(numBytes);

    // Calculem el producte estàndard a la CPU per comparar
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        cpuC[i*N+j] = 0.0;
        for (int k = 0; k < N; k++)
          cpuC[i*N+j] += hA[i*N+k] * hB[k*N+j];
      }
    }

    // Comparem els resultats (verificació parcial per matrius grans)
    bool correct = true;
    int errors = 0;
    int maxErrors = 10; // Limitem el nombre d'errors mostrats

    for (int i = 0; i < N && errors < maxErrors; i += N/16) { // Mostreig cada N/16 files
      for (int j = 0; j < N && errors < maxErrors; j += N/16) { // Mostreig cada N/16 columnes
        if (fabs(cpuC[i*N+j] - hC[i*N+j]) > 0.001) {
          correct = false;
          printf("Error a la posició (%d,%d): CPU=%f, GPU=%f, diff=%f\n", 
              i, j, cpuC[i*N+j], hC[i*N+j], fabs(cpuC[i*N+j] - hC[i*N+j]));
          errors++;
        }
      }
    }

    if (correct)
      printf("TEST PASS\n");
    else
      printf("TEST FAIL (%d errors trobats en mostreig)\n", errors);

    free(cpuC);
  }

  // Alliberem memòria del host
  cudaFreeHost(hA); cudaFreeHost(hB); cudaFreeHost(hC);
  cudaFreeHost(hA11); cudaFreeHost(hA12); cudaFreeHost(hA21); cudaFreeHost(hA22);
  cudaFreeHost(hB11); cudaFreeHost(hB12); cudaFreeHost(hB21); cudaFreeHost(hB22);
  cudaFreeHost(hC11); cudaFreeHost(hC12); cudaFreeHost(hC21); cudaFreeHost(hC22);

  return 0;
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
