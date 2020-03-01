#include <stdio.h>
#include<cuda.h>
#include<iostream>
#include<stdio.h>
#include<cuda.h>
#include "DotProduct.h"

int main (int argc, char* argv[]) {

	//declare user input
	int rows, cols, threads, CUDA_DEVICE;
	//char *matrix, *w_vectorF;
	
	//take user arguments to parameters	
	rows = atoi(argv[1]);
	cols = atoi(argv[2]);
        //matrix = atoi(argv[3]);
        //w_vectorF = atoi(argv[4]);
	CUDA_DEVICE = atoi(argv[5]);
	threads = atoi(argv[6]);
   
	//declare pointers
        FILE *fp, *fv;

	//declare the size variable for data
	size_t data_size;
	size_t vec_Size;

	//calculate the totle size of the training data
	data_size = (size_t)((size_t)rows * (size_t)cols);

	//calculate the totol size of the weight vector data
	vec_Size = ((size_t)cols);
	

	//give space to w_vector data in cpu
	float *w_vect=(float*)malloc((vec_Size)*sizeof(float));

	//give space to train data in cpu
	float *host_train = (float*)malloc((data_size)*sizeof(float));
 
	//give space to result from device in cpu
        float* host_partition = (float*) malloc(rows*sizeof(float));

	//declare variables for spacing to GPU
        float *dev_w_vect, *dev_train, *dev_partition;
	
	//for reading data from the file to cpu
        float file_data;
	float mat[rows][cols];

	//output error if cuda device is not working
	cudaError err = cudaSetDevice(CUDA_DEVICE);
	if(err != cudaSuccess) { printf("Error setting CUDA DEVICE\n"); exit(EXIT_FAILURE); }

	//make space for gpu
	cudaMalloc((float**)&dev_w_vect, vec_Size*sizeof(float));
	cudaMalloc((float**)&dev_train, data_size*sizeof(float));
	cudaMalloc((float**)&dev_partition, rows*sizeof(float));
	
        //open train folder, output error if is not working
        fp = fopen(argv[3], "r");
	   if (fp == NULL) {
    		printf("Cannot Open the File");
		return 0;
	}

    
	//read train data in the order, map to matrix form
        int i=0;
        int j = 0;
	for(i = 0; i < rows; i++)
 	{
      		for(j = 0; j < cols; j++) 
      		{
			fscanf(fp, "%f", &file_data);
			mat[i][j] = file_data;		   
		}
	 }
	  fclose(fp);
	
	//save the data to host memory host_train
	for(int i= 0; i < cols; i++)
	  {
		for(int j = 0; j < rows; j++)
		  {   
			  host_train[rows*i+j] = mat[j][i];
		  }
	
	  }
	//open w vector file
	fv = fopen(argv[4], "r");
	if (fv == NULL) {
                printf("Cannot Open the File");
                return 0;
        }
	//read vector from w_vector file	 	  
	for(int j = 0; j < cols; j++) 
	{
		 fscanf(fv, "%f", &w_vect[j]);
	}
	  
	  fclose(fv);
		
	//host to device data transfer for both vector and traindata
	cudaMemcpy(dev_w_vect, w_vect, vec_Size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_train, host_train, data_size*sizeof(float), cudaMemcpyHostToDevice);
	
	//get user input thread; calculate block
	int jobs = cols;
	int BLOCKS = (jobs + threads - 1)/threads;

	//start kernel
	kernel<<<BLOCKS, threads>>>(dev_w_vect, dev_train, dev_partition, rows, cols);

	//get the result back from gpu to cpu
	cudaMemcpy(host_partition,dev_partition, rows*sizeof(float), cudaMemcpyDeviceToHost);
	
	//write result
	for(int i=0; i<rows; i++) {
        	printf("%f\n", host_partition[i]);
    	}
    

	cudaFree(dev_w_vect);
	cudaFree(dev_train);
	cudaFree(dev_partition);
	free(host_partition);
}
