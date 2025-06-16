#!/bin/bash

### Directivas para el gestor de colas
#SBATCH --job-name=MultiGPU
#SBATCH -D .
#SBATCH --output=submit-MultiGPU.o%j
#SBATCH --error=submit-MultiGPU.e%j
#SBATCH -A cuda
#SBATCH -p cuda
## OPCION B: Usamos las 4 RTX 3080
#SBATCH --qos=cuda3080  
#SBATCH --gres=gpu:rtx3080:4


export PATH=/Soft/cuda/12.2.2/bin:$PATH

# Para comprobar que funciona no es necesario usar matrices muy grandes
# Con N = 1024 es suficiente

./STR1GPUs.exe       1024 N
./STR1GPUs.exe       2048 N
./STR1GPUs.exe       4096 N
./STR1GPUs.exe       8192 N
./STR1GPUs.exe      16384 N

./STR1GPUspin.exe    1024 N
./STR1GPUspin.exe    2048 N
./STR1GPUspin.exe    4096 N
./STR1GPUspin.exe    8192 N
./STR1GPUspin.exe   16384 N

./STR1GPUsasync.exe  1024 N
./STR1GPUsasync.exe  2048 N
./STR1GPUsasync.exe  4096 N
./STR1GPUsasync.exe  8192 N
./STR1GPUsasync.exe 16384 N

#nsys nvprof --print-gpu-trace ./kernel4GPUs.exe 2048 N

#nsys nvprof --print-gpu-trace ./kernel4GPUs.exe 1024 N








