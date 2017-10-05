#################################################
##   Makefile for spectral graph partitioning  ##
##   Create by Wei-Chien Liao                  ##
##   Modified by Yuhsiang Tsai				   ##
##   Date  : 2017/09/04                        ##
##   Update: 2017/10/04                        ##
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
NVCCFLAGS	= -m64 -std=c++11

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
CUDAINCS  = -I$(CUDAROOT)/include
CUDALIBS  = -L$(CUDAROOT)/lib64
CUDALNKS  = -lcublas -lcusparse -lcudart -lcusolver

#MAGMA
MAGMAROOT ?= /opt/magma/2.2f
MAGMAINCS  = -isystem $(MAGMAROOT)/include
MAGMALIBS  = -L$(MAGMAROOT)/lib
MAGMALNKS  = -lmagma_sparse -lmagma
#==============================================================

obj = map_boundary.o verify_boundary_sparse.o write_object.o \
	read_args.o read_object.o reorder_vertex.o \
    construct_laplacian_sparse.o solve_shiftevp_cuda.o \
    solve_harmonic_sparse.o
	  
magma_obj = map_boundary.o verify_boundary_sparse.o write_object.o \
	read_args.o read_object.o reorder_vertex.o \
    construct_laplacian_sparse.o solve_shiftevp_cuda.o \
    solve_harmonic_sparse_magma.o

sgp_obj = read_graph.o graph_adjacency.o graph_laplacian.o \
	solve_shiftevp_cuda.o set_graph_type.o 

INCS = -I include

.PHONY: all clean

all: main_3Dface_evp.out sgp_main.out


%.o: %.cpp
	$(CC) $(CCFLAGS) -c $< $(INCS)  $(MKLINCS)

%.o: src/core/%.cpp
	$(CC) $(CCFLAGS) -c $< $(INCS) $(MKLINCS) $(CUDAINCS)

%.o: src/cuda/%.cpp
	$(CC) $(CCFLAGS) -c $< $(INCS) $(MKLINCS) $(CUDAINCS)

%.o: src/sparse/%.cpp
	$(CC) $(CCFLAGS) -c $< $(INCS) $(MKLINCS) $(CUDAINCS)

%.o: src/magma/%.cpp
	$(CC) $(CCFLAGS) -c $< $(INCS) $(MKLINCS) $(CUDAINCS) $(MAGMAINCS)

MakeExe:sgp_main.out main_3Dface_evp.out

sgp_main.out: sgp_main.o $(sgp_obj)
	$(LOADER) $(CCFLAGS) -o $@ $< $(sgp_obj) $(MKLLNKS) $(CUDALNKS)

main_3Dface_evp.out: main_3Dface_evp.o $(obj)
	$(LOADER) $(CCFLAGS) -o $@ $< $(obj) $(MKLLNKS) $(CUDALNKS)

magma_3Dface_evp.out: main_3Dface_evp.o $(magma_obj)
	$(LOADER) $(CCFLAGS) -o $@ $< $(magma_obj) $(MKLLIBS) $(CUDALIBS) $(MAGMALIBS) $(MKLLNKS) $(CUDALNKS) $(MAGMALNKS)

clean:
	-rm *.o -f
	-rm *.out -f