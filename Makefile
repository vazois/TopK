CC=g++
NVCC=nvcc

CC_MAIN=cpu/main.cpp
CC_FLAGS=-std=c++11
CC_EXE=cpu_run
CC_OPT_FLAGS= -O3 -ffast-math -funroll-loops -msse -msse3 -mmmx -fomit-frame-pointer -m64

all: cpu_cc

cpu_cc: 
	$(CC) $(CC_FLAGS) $(CC_OPT_FLAGS) $(CC_MAIN) -o $(CC_EXE)
	
clean:
	rm -rf $(CC_EXE)  
