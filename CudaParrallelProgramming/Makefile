CXX = nvcc
CXX2 = gcc
CXXFLAGS2 = -O3 -Wall
#CXXFLAGS3 = -arch=sm_20 -use_fast_math -O3
CXXFLAGS3 = -O3 
#CXXFLAGS3 =
TARGET1= DotProduct

all : $(TARGET1)
    
$(TARGET1) : DotProduct.cu kernel.cu DotProduct.h
	$(CXX) $(CXXFLAGS3) -lm -o $(TARGET1) DotProduct.cu kernel.cu
clean : 
	rm -f $(TARGET1)
