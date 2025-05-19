CUDA_HOME   = /Soft/cuda/12.2.2
NVCC        = $(CUDA_HOME)/bin/nvcc
ARCH       = -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89
NVCC_FLAGS  = -O3 -Wno-deprecated-gpu-targets -I$(CUDA_HOME)/include $(ARCH) --ptxas-options=-v -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
LD_FLAGS    = -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib

PROG_FLAGS  = -DPINNED=0 -DSIZE=32

EXE00	        = kernel00.exe
EXE01	        = kernel01.exe
EXE10	        = kernel10.exe
EXE11	        = kernel11.exe
EXE22	        = kernel22.exe

OBJ00	        = MM00.o
OBJ01	        = MM01.o
OBJ10	        = MM10.o
OBJ11	        = MM11.o
OBJ22	        = MM22.o

default: $(EXE00)

MM00.o: MM00.cu
	$(NVCC) -c -o $@ MM00.cu $(NVCC_FLAGS) $(PROG_FLAGS)
MM01.o: MM01.cu
	$(NVCC) -c -o $@ MM01.cu $(NVCC_FLAGS) $(PROG_FLAGS)
MM10.o: MM10.cu
	$(NVCC) -c -o $@ MM10.cu $(NVCC_FLAGS) $(PROG_FLAGS)
MM11.o: MM11.cu
	$(NVCC) -c -o $@ MM11.cu $(NVCC_FLAGS) $(PROG_FLAGS)
MM22.o: MM22.cu
	$(NVCC) -c -o $@ MM22.cu $(NVCC_FLAGS) $(PROG_FLAGS)


$(EXE00): $(OBJ00)
	$(NVCC) $(OBJ00) -o $(EXE00) $(LD_FLAGS)

$(EXE01): $(OBJ01)
	$(NVCC) $(OBJ01) -o $(EXE01) $(LD_FLAGS)

$(EXE10): $(OBJ10)
	$(NVCC) $(OBJ10) -o $(EXE10) $(LD_FLAGS)

$(EXE11): $(OBJ11)
	$(NVCC) $(OBJ11) -o $(EXE11) $(LD_FLAGS)

$(EXE22): $(OBJ22)
	$(NVCC) $(OBJ22) -o $(EXE22) $(LD_FLAGS)


all:	$(EXE00) $(EXE01) $(EXE10) $(EXE11) $(EXE22)

clean:
	rm -rf *.o kernel*.exe

