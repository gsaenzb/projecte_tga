#!/bin/bash

### Directivas per al gestor de cues
#SBATCH --job-name=StrassenMM
#SBATCH -D .
#SBATCH --output=submit-StrassenMM.o%j
#SBATCH --error=submit-StrassenMM.e%j
#SBATCH -A cuda
#SBATCH -p cuda

## NOMÉS 1 DE LES TRES OPCIONS POT ESTAR ACTIVA
## OPCIÓ A: Utilitzem la RTX 4090
##SBATCH --qos=cuda4090  
##SBATCH --gres=gpu:rtx4090:1

## OPCIÓ B: Utilitzem les 4 RTX 3080
##SBATCH --qos=cuda3080  
##SBATCH --gres=gpu:rtx3080:4

## OPCIÓ C: Utilitzem 1 RTX 3080
#SBATCH --qos=cuda3080  
#SBATCH --gres=gpu:rtx3080:1

export PATH=/Soft/cuda/12.2.2/bin:$PATH

#$ -N StrassenMM

# Execucions amb mides diferents de matriu i comprovació de resultats
#./kernel10.exe 128 Y
#./kernel10.exe 256 Y
#./kernel10.exe 512 Y
#./kernel10.exe 1024 Y
./kernel10.exe 2048 Y

# Execucions per mesurar rendiment sense comprovació
#./kernel10.exe 128 N
#./kernel10.exe 256 N
#./kernel10.exe 512 N
#./kernel10.exe 1024 N
#./kernel10.exe 2048 N
#./kernel10.exe 4096 N

# Opcional: comparació amb el mètode original (si és disponible)
# ./MM10.exe 1024 1024 1024 N

# Opcional: Execucions amb NVIDIA Compute Profiler per analitzar rendiment
# ncu --set full ./kernel10.exe 1024 N
