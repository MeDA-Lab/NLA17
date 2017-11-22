#################################################
##   Makefile for spectral graph partitioning  ##
##   Create by Wei-Chien Liao                  ##
##   Modified by Yuhsiang Tsai				   ##
##   Date  : 2017/09/04                        ##
##   Update: 2017/10/12                        ##
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
CCFLAGS		= -O3 -m64 -std=c++11 -fopenmp
NVCCFLAGS	= -std=c++11

#======================
# Library PATH settings
#======================
#MKL
# MKLROOT = /opt/intel/mkl
MKLINCS = -m64 -I$(MKLROOT)/include
MKLLIBS = -L$(MKLROOT)/lib/intel64
MKLLNKS =  -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

#CUDA
CUDAROOT ?= /usr/local/cuda-8.0
CUDAINCS  = -I$(CUDAROOT)/include -I$(CUDAROOT)/samples/common/inc
CUDALIBS  = -L$(CUDAROOT)/lib64
CUDALNKS  = -lcublas -lcusparse -lcudart -lcusolver

#MAGMA
MAGMAROOT ?= /opt/magma/2.2f
MAGMAINCS  = -isystem $(MAGMAROOT)/include
MAGMALIBS  = -L$(MAGMAROOT)/lib
MAGMALNKS  = -lmagma_sparse -lmagma
#==============================================================

surfpara_obj = map_boundary.o verify_boundary_sparse.o write_object.o \
			   read_args.o read_object.o reorder_vertex.o string2arg.o\
    		   construct_laplacian_sparse.o solve_shiftevp_cuda.o \
               magma_dcsrset_gpu.o solve_harmonic_sparse.o
	  
surfpara_magma_obj = map_boundary.o verify_boundary_sparse.o write_object.o \
					 read_args.o read_object.o reorder_vertex.o string2arg.o\
    				 construct_laplacian_sparse.o solve_shiftevp_cuda.o \
    				 magma_dcsrset_gpu.o solve_harmonic_sparse_magma.o \
    				 invLanczos_magma_gpu.o Lanczos_decomp_gpu.o \
	Lanczos_LockPurge_gpu.o GVqrrq_g.o

sgp_obj = read_graph.o graph_adjacency.o graph_laplacian.o \
	solve_shiftevp_cuda.o generate_RHS.o \
	sgp_read_args.o solve_sgp_sparse_magma.o \
	cuda_solverinfo.o residual.o solve_ls_cuda.o \
	lu_cuda.o chol_cuda.o qr_cuda.o string2arg.o print_konect_header.o \
	solve_SMEVP.o magma_dcsrset_gpu.o write_partition.o \
	invLanczos_magma_gpu.o Lanczos_decomp_gpu.o \
	Lanczos_LockPurge_gpu.o GVqrrq_g.o


INCS = -I include

.PHONY: all clean

all: MakeExe


%.o: %.cpp
	$(CC) $(CCFLAGS) -c $< $(INCS)  $(MKLINCS)

%.o: src/core/%.cpp
	$(CC) $(CCFLAGS) -c $< $(INCS) $(MKLINCS) $(CUDAINCS)

%.o: src/tool/%.cpp
	$(CC) $(CCFLAGS) -c $< $(INCS) $(MKLINCS) $(CUDAINCS)

%.o: src/cuda/%.cpp
	$(CC) $(CCFLAGS) -c $< $(INCS) $(MKLINCS) $(CUDAINCS)

%.o: src/sparse/%.cpp
	$(CC) $(CCFLAGS) -c $< $(INCS) $(MKLINCS) $(CUDAINCS)

%.o: src/magma/%.cpp
	$(CC) $(CCFLAGS) -c $< $(INCS) $(MKLINCS) $(CUDAINCS) $(MAGMAINCS)

%.o: src/magma/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< $(INCS) $(MKLINCS) $(CUDAINCS) $(MAGMAINCS)

MakeExe: surfpara_magma.out sgp.out

sgp.out: sgp.o $(sgp_obj)
	$(LOADER) $(CCFLAGS) -o $@ $< $(sgp_obj) $(MAGMALIBS) $(MAGMALNKS) $(MKLLNKS) $(CUDALNKS)

surfpara.out: surfpara.o $(surfpara_obj)
	$(LOADER) $(CCFLAGS) -o $@ $< $(surfpara_obj) $(MKLLNKS) $(CUDALNKS)

surfpara_magma.out: surfpara.o $(surfpara_magma_obj)
	$(LOADER) $(CCFLAGS) -o $@ $< $(surfpara_magma_obj) $(MKLLIBS) $(CUDALIBS) $(MAGMALIBS) $(MKLLNKS) $(CUDALNKS) $(MAGMALNKS)

clean:
	-rm *.o -f
	-rm *.out -f