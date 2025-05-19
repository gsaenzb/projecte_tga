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

// C2,2 = M1 * M2 + M3 + M6
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

  float *hA, *hB, *hC, *hCCheck;
  float *dA, *dB, *dC;
  float *dA11, *dA12, *dA21, *dA22;
  float *dB11, *dB12, *dB21, *dB22;
  float *dC11, *dC12, *dC21, *dC22;
  float *dM1, *dM2, *dM3, *dM4, *dM5, *dM6, *dM7;

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

  // Obtenim memòria al host
  cudaMallocHost((float**)&hA, numBytes);
  cudaMallocHost((float**)&hB, numBytes);
  cudaMallocHost((float**)&hC, numBytes);
  cudaMallocHost((float**)&hCCheck, numBytes);

  // Inicialitzem les matrius
  InitM(N, N, hA);
  InitM(N, N, hB);

  // Obtenim memòria al device
  cudaMalloc((float**)&dA, numBytes);
  cudaMalloc((float**)&dB, numBytes);
  cudaMalloc((float**)&dC, numBytes);

  // Memòria per a les submatrius A i B
  cudaMalloc((float**)&dA11, numBytesHalf);
  cudaMalloc((float**)&dA12, numBytesHalf);
  cudaMalloc((float**)&dA21, numBytesHalf);
  cudaMalloc((float**)&dA22, numBytesHalf);
  cudaMalloc((float**)&dB11, numBytesHalf);
  cudaMalloc((float**)&dB12, numBytesHalf);
  cudaMalloc((float**)&dB21, numBytesHalf);
  cudaMalloc((float**)&dB22, numBytesHalf);

  // Memòria per a les submatrius C
  cudaMalloc((float**)&dC11, numBytesHalf);
  cudaMalloc((float**)&dC12, numBytesHalf);
  cudaMalloc((float**)&dC21, numBytesHalf);
  cudaMalloc((float**)&dC22, numBytesHalf);

  // Memòria per a les matrius M de Strassen
  cudaMalloc((float**)&dM1, numBytesHalf);
  cudaMalloc((float**)&dM2, numBytesHalf);
  cudaMalloc((float**)&dM3, numBytesHalf);
  cudaMalloc((float**)&dM4, numBytesHalf);
  cudaMalloc((float**)&dM5, numBytesHalf);
  cudaMalloc((float**)&dM6, numBytesHalf);
  cudaMalloc((float**)&dM7, numBytesHalf);

  cudaEventCreate(&E0);
  cudaEventCreate(&E1);
  cudaEventCreate(&E2);
  cudaEventCreate(&E3);

  // Copiem les matrius A i B al device
  cudaMemcpy(dA, hA, numBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, numBytes, cudaMemcpyHostToDevice);

  cudaEventRecord(E0, 0);

  // Iniciem el cronòmetre per mesurar el temps dels kernels
  cudaEventRecord(E1, 0);

  // Implementem l'algorisme de Strassen

  // 1. Dividim les matrius A i B en submatrius
  // Aquí caldria implementar kernels per dividir les matrius, però per simplificar
  // Es podria fer manualment a la CPU i transferir les dades

  // Per exemple, per a la matriu A:
  int half_N = N/2;
  for (int i = 0; i < half_N; i++) {
    cudaMemcpy(&dA11[i*half_N], &hA[i*N], half_N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&dA12[i*half_N], &hA[i*N+half_N], half_N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&dA21[i*half_N], &hA[(i+half_N)*N], half_N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&dA22[i*half_N], &hA[(i+half_N)*N+half_N], half_N*sizeof(float), cudaMemcpyHostToDevice);
  }

  // I per a la matriu B:
  for (int i = 0; i < half_N; i++) {
    cudaMemcpy(&dB11[i*half_N], &hB[i*N], half_N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&dB12[i*half_N], &hB[i*N+half_N], half_N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&dB21[i*half_N], &hB[(i+half_N)*N], half_N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&dB22[i*half_N], &hB[(i+half_N)*N+half_N], half_N*sizeof(float), cudaMemcpyHostToDevice);
  }

  // 2. Calculem les 7 matrius M utilitzant els kernels que hem creat
  KernelM1<<<dimGrid, dimBlock>>>(half_N, dA11, dA22, dB11, dB22, dM1);
  KernelM2<<<dimGrid, dimBlock>>>(half_N, dA21, dA22, dB11, dM2);
  KernelM3<<<dimGrid, dimBlock>>>(half_N, dA11, dB12, dB22, dM3);
  KernelM4<<<dimGrid, dimBlock>>>(half_N, dA22, dB21, dB11, dM4);
  KernelM5<<<dimGrid, dimBlock>>>(half_N, dA11, dA12, dB22, dM5);
  KernelM6<<<dimGrid, dimBlock>>>(half_N, dA21, dA11, dB11, dB12, dM6);
  KernelM7<<<dimGrid, dimBlock>>>(half_N, dA12, dA22, dB21, dB22, dM7);

  // 3. Calculem les submatrius C
  KernelC11<<<dimGrid, dimBlock>>>(half_N, dM1, dM4, dM5, dM7, dC11);
  KernelC12<<<dimGrid, dimBlock>>>(half_N, dM3, dM5, dC12);
  KernelC21<<<dimGrid, dimBlock>>>(half_N, dM2, dM4, dC21);
  KernelC22<<<dimGrid, dimBlock>>>(half_N, dM1, dM2, dM3, dM6, dC22);

  // Finalitzem el cronòmetre dels kernels
  cudaEventRecord(E2, 0);
  cudaEventSynchronize(E2);

  // 4. Unir les submatrius C per obtenir el resultat final
  // Copiem les dades a la matriu C completa
  for (int i = 0; i < half_N; i++) {
    cudaMemcpy(&hC[i*N], &dC11[i*half_N], half_N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hC[i*N+half_N], &dC12[i*half_N], half_N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hC[(i+half_N)*N], &dC21[i*half_N], half_N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hC[(i+half_N)*N+half_N], &dC22[i*half_N], half_N*sizeof(float), cudaMemcpyDeviceToHost);
  }

  cudaEventRecord(E3, 0);
  cudaEventSynchronize(E3);

  // Alliberem memòria al device
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  cudaFree(dA11); cudaFree(dA12); cudaFree(dA21); cudaFree(dA22);
  cudaFree(dB11); cudaFree(dB12); cudaFree(dB21); cudaFree(dB22);
  cudaFree(dC11); cudaFree(dC12); cudaFree(dC21); cudaFree(dC22);
  cudaFree(dM1); cudaFree(dM2); cudaFree(dM3); cudaFree(dM4);
  cudaFree(dM5); cudaFree(dM6); cudaFree(dM7);

  cudaEventElapsedTime(&TiempoTotal, E0, E3);
  cudaEventElapsedTime(&TiempoKernel, E1, E2);

  printf("\nKERNEL Strassen 1 GPU - Producto Matrices\n");
  printf("Dimensiones: %dx%d\n", N, N);
  printf("nThreads: %dx%d (%d)\n", nThreads, nThreads, nThreads * nThreads);
  printf("nBlocks: %dx%d (%d)\n", nBlocks, nBlocks, nBlocks*nBlocks);
  printf("Tiempo Global: %4.6f milseg\n", TiempoTotal);
  printf("Tiempo Kernel: %4.6f milseg\n", TiempoKernel);
  printf("Rendimiento Global: %4.2f GFLOPS\n", (2.0 * (float) N * (float) N * (float) N) / (1000000.0 * TiempoTotal));
  printf("Rendimiento Kernel: %4.2f GFLOPS\n", (2.0 * (float) N * (float) N * (float) N) / (1000000.0 * TiempoKernel));

  cudaEventDestroy(E0); cudaEventDestroy(E1); cudaEventDestroy(E2); cudaEventDestroy(E3);

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

