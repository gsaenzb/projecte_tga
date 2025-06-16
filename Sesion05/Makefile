CUDA_HOME   = /Soft/cuda/12.2.2
NVCC        = $(CUDA_HOME)/bin/nvcc
ARCH       = -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89
NVCC_FLAGS  = -O3 -Wno-deprecated-gpu-targets -I$(CUDA_HOME)/include $(ARCH) --ptxas-options=-v -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
LD_FLAGS    = -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib

PROG_FLAGS  = -DSIZE=32

EXE = STR1GPUs.exe STR1GPUspin.exe STR1GPUsasync.exe STR2GPUs.exe STR4GPUs.exe
OBJ = STR1GPUs.o STR1GPUspin.o STR1GPUsasync.o STR2GPUs.o STR4GPUs.o

default: all

STR1GPUs.o: STR1GPUs.cu
	$(NVCC) -c -o $@ STR1GPUs.cu $(NVCC_FLAGS) $(PROG_FLAGS)

STR1GPUs.exe: STR1GPUs.o
	$(NVCC) STR1GPUs.o -o STR1GPUs.exe $(LD_FLAGS)

STR1GPUspin.o: STR1GPUs.cu
	$(NVCC) -c -o $@ STR1GPUs.cu -DPINNED_MEMORY $(NVCC_FLAGS) $(PROG_FLAGS)

STR1GPUspin.exe: STR1GPUspin.o
	$(NVCC) STR1GPUspin.o -o STR1GPUspin.exe -DPINNED_MEMORY $(LD_FLAGS)

STR1GPUsasync.o: STR1GPUs.cu
	$(NVCC) -c -o $@ STR1GPUs.cu -DPINNED_MEMORY -DASYNC_CALLS $(NVCC_FLAGS) $(PROG_FLAGS)

STR1GPUsasync.exe: STR1GPUsasync.o
	$(NVCC) STR1GPUsasync.o -o STR1GPUsasync.exe -DPINNED_MEMORY -DASYNC_CALLS $(LD_FLAGS)

STR2GPUs.o: STR2GPUs.cu
	$(NVCC) -c -o $@ STR2GPUs.cu $(NVCC_FLAGS) $(PROG_FLAGS)

STR2GPUs.exe: STR2GPUs.o
	$(NVCC) STR2GPUs.o -o STR2GPUs.exe $(LD_FLAGS)

STR4GPUs.o: STR4GPUs.cu
	$(NVCC) -c -o $@ STR4GPUs.cu $(NVCC_FLAGS) $(PROG_FLAGS)

STR4GPUs.exe: STR4GPUs.o
	$(NVCC) STR4GPUs.o -o STR4GPUs.exe $(LD_FLAGS)

all: $(EXE) 

clean:
	rm -rf *.o kernel*.exe

