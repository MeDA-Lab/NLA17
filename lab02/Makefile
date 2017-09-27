#################################################
##   Makefile for spectral graph partitioning  ##
##   Create by Wei-Chien Liao                  ##
##   Date  : 2017/09/04                        ##
##   Update:                                   ##
#################################################

#=======================
# Compiler settings
#=======================
CC          = g++
NVCC        = nvcc
LOADER      = $(CC)

ARCH        = ar
ARCHFLAGS   = cr
RANLIB      = ranlib

#=======================
# Flag settings.
#=======================
CCFLAGS		= -O3 -m64 -std=c++11
NVCCFLAGS	= -m64 -std=c++11

#======================
# Library PATH settings
#======================
#MKL
MKLROOT = /opt/intel/mkl
MKLINCS = -I${MKLROOT}/include
MKLLNKS = -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

#CUDA
CUDA_PATH = /usr/local/cuda-8.0
CUDA_INC  = -I$(CUDA_PATH)/include
CUDA_LIB_PATH   = $(CUDA_PATH)/lib64
CUDA_LD_FLAGS   = -lcudart -lcublas -lcufft -lcusolver -lcusparse -lgomp -lm -ldl
CUDA_LD_LIB	    = -L$(CUDA_LIB_PATH) $(CUDA_LD_FLAGS)

#==============================================================

obj = read_graph.o graph_adjacency.o graph_laplacian.o \
      solve_shiftevp_cuda.o map_boundary.o \
      read_args.o read_object.o reorder_vertex.o \
      construct_laplacian_sparse.o solve_harmonic_sparse.o \
      verify_boundary_sparse.o set_graph_type.o \
      residual.o generate_RHS.o \
      solve_ls_cuda.o lu_cuda.o qr_cuda.o chol_cuda.o

INCS = -I include
TARGETS_O	:= $(TARGETS_SRC:.cpp=.o)

.PHONY: all clean

all: MakeObj MakeExe

MakeObj: $(TARGETS_O)

%.o: %.cpp
	$(CC) -c $< $(INCS) $(CCFLAGS) $(MKLINCS)

%.o: src/core/%.cpp
	$(CC) -c $< $(INCS) $(CCFLAGS) $(MKLINCS) $(CUDA_INC)

%.o: src/cuda/%.cpp
	$(CC) -c $< $(INCS) $(CCFLAGS) $(MKLINCS) $(CUDA_INC)

%.o: src/sparse/%.cpp
	$(CC) -c $< $(INCS) $(CCFLAGS) $(MKLINCS) $(CUDA_INC)

MakeExe:sgp_main_ls.out main_3Dface_ls.out

sgp_main_ls.out: sgp_main_ls.o $(obj)
	$(LOADER) $< -o $@ $(obj) $(CCFLAGS) $(MKLLNKS) $(CUDA_LD_FLAGS)

main_3Dface_ls.out: main_3Dface_ls.o $(obj)
	$(LOADER) $< -o $@ $(obj) $(CCFLAGS) $(MKLLNKS) $(CUDA_LD_FLAGS)

clean:
	-rm *.o -f
	-rm *.out -f