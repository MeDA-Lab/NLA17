# NLA17 - lab01
This is lab01 of NLA17.  
You will develop a program which compute dot product of two vectors.  

# Equipment
- INTEL-MKL
- CUDA

# For this course
If you use the workstation which we provide in this course, you can follow 
the following step to finish lab01.
## module system
We build up a module system for using library easily.
```
module load intel-mkl
module load cuda-dev
```
You can use __MKL__ and __CUDA__ library now.
## Compile the code
```
g++ lab01.cpp -c  -m64 -I${MKLROOT}/include -I${CUDADIR}/include -std=c++11
g++ lab01.o -o lab01  -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed 
-lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl 
-L${CUDADIR}/lib64 -lcudart -lcublas
```
## Run the code
```
./lab01
```
