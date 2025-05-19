CUDA_HOME   = /Soft/cuda/12.2.2
NVCC        = $(CUDA_HOME)/bin/nvcc
ARCH       = -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89
NVCC_FLAGS  = -O3 -Wno-deprecated-gpu-targets -I$(CUDA_HOME)/include $(ARCH) --ptxas-options=-v -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
LD_FLAGS    = -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib

PROG_FLAGS  = -DSIZE=32


EXE4GPUs    = kernel4GPUs.exe

OBJ4GPUs    = MM4GPUs.o

default: $(EXE4GPUs)

MM4GPUs.o: MM4GPUs.cu
	$(NVCC) -c -o $@ MM4GPUs.cu $(NVCC_FLAGS) $(PROG_FLAGS)


$(EXE4GPUs): $(OBJ4GPUs)
	$(NVCC) $(OBJ4GPUs) -o $(EXE4GPUs) $(LD_FLAGS)


all:	$(EXE4GPUs) 

clean:
	rm -rf *.o kernel*.exe

