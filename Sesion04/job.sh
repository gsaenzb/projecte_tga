#!/bin/bash

### Directivas para el gestor de colas
#SBATCH --job-name=ProdMbyM
#SBATCH -D .
#SBATCH --output=submit-ProdMbyM.o%j
#SBATCH --error=submit-ProdMbyM.e%j
#SBATCH -A cuda
#SBATCH -p cuda

## SOLO 1 DE LAS TRES OPCIONES PUEDE ESTAR ACTIVA
## OPCION A: Usamos la RTX 4090
##SBATCH --qos=cuda4090  
##SBATCH --gres=gpu:rtx4090:1

## OPCION B: Usamos las 4 RTX 3080
##SBATCH --qos=cuda3080  
##SBATCH --gres=gpu:rtx3080:4

## OPCION C: Usamos 1 RTX 3080
#SBATCH --qos=cuda3080  
#SBATCH --gres=gpu:rtx3080:1



export PATH=/Soft/cuda/12.2.2/bin:$PATH


#$ -N ProdMbyM 

./kernel00.exe  640 Y
./kernel00.exe  641 Y

#./kernel01.exe 639 641 1023 Y
#./kernel10.exe 640 512 1024 Y

#./kernel10.exe 639 641 1023 Y
#./kernel11.exe 640 512 1024 Y
#./kernel11.exe 639 641 1023 Y


#ncu --set full ./kernel01.exe 2048 2048 2048 N


