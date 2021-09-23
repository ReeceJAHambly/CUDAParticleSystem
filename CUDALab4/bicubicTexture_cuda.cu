/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


#ifndef _BICUBICTEXTURE_CU_
#define _BICUBICTEXTURE_CU_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <helper_math.h>

#include <helper_cuda.h>
#include "hitable.h"
#include "hitable_list.h"
#include "ray.h"
#include "sphere.h"
#include "vec3.h"
#include <curand.h>
#include <curand_kernel.h>
#include "math.h"
#include "cuda_runtime.h"



typedef unsigned int uint;
typedef unsigned char uchar;

//#include "bicubicTexture_kernel.cuh"

cudaArray *d_imageArray = 0;
cudaEvent_t start, stop;
float elapsed_time_ms;

int const numOfParticles = 100;
__device__ float static ticks = 0;
__device__ float xWindPos = 0;

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << "at " << file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

__device__ vec3 castRay(const ray& r, hitable **world) {
	hit_record rec;
	if ((*world)->hit(r, 0.0, FLT_MAX, rec)) {
		return 0.5f*vec3(rec.normal.x() + 1.0f, rec.normal.y() + 1.0f,
			rec.normal.z() + 1.0f);
	}
	else {
		vec3 unit_direction = unit_vector(r.direction());
		float t = 0.5f*(unit_direction.y() + 1.0f);
		return (1.0f - t)*vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
	}
}

__global__ void setup_kernel(curandState * state, unsigned long seed)
{
	int id = threadIdx.x;
	curand_init(seed, id, 0, &state[id]);
}

__device__ float generate(curandState* globalState, int ind)
{
	curandState localState = globalState[ind];
	float RANDOM = curand_uniform(&localState);
	globalState[ind] = localState;
	return RANDOM;
}

__global__ void create_world(hitable **d_list, hitable **d_world) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {

		d_list[0] = new sphere(vec3(0, 0, -10001.0), 10000);
		d_list[1] = new sphere(vec3(-10002.0, 0, -3), 10000);
		d_list[2] = new sphere(vec3(10002.0, 0, -3), 10000);
		d_list[3] = new sphere(vec3(0, 10002.0, -3), 10000);
		d_list[4] = new sphere(vec3(0, -10002.0, -3), 10000);

		*d_world = new hitable_list(d_list, 5);
	}
}

__global__ void create_balls(hitable **d_list, hitable **d_world, float* xPos,float*yPos, curandState* globalState) {
	int x = threadIdx.x;

	float numberX = (2.0 * generate(globalState, 5 + x)) - 1.0;
	float numberY = (2.0 * generate(globalState, 2 + x)) - 1.0;

	if (xPos[x] > 2.0) {
		xPos[x] = -2.0;
	}
	if (xPos[x] < -2.0) {
		xPos[x] = 2.0;
	}
	xPos[x] += (numberX * 0.01);
		
	if (yPos[x] > 2.0) {
		yPos[x] = -2.0;
	}
	if (yPos[x] < -2.0) {
		yPos[x] = 2.0;
	}
	yPos[x] += (numberY * 0.01);
	
	d_list[5+x] = new sphere(vec3(xPos[x], yPos[x], -1.0), 0.05);
	*d_world = new hitable_list(d_list,5 + numOfParticles);
}


__global__ void windKernel(float* xPos) {
	int x = threadIdx.x;
	float windX = 0.1;
	xPos[x] += windX;
}

__global__ void gravityKernel(float*yPos) {
	int x = threadIdx.x;
	float Gravity = -0.098;
	yPos[x] += Gravity;
}



__global__  void free_world(hitable **d_list, hitable **d_world) {
	delete *(d_list);
	delete *d_world;
}

__global__ void d_render(uchar4 *d_output, uint width, uint height, hitable ** d_world)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint i = y * width + x;

	float u = x / (float)width; //----> [0, 1]x[0, 1]
	float v = y / (float)height;
	u = 2.0*u - 1.0; //---> [-1, 1]x[-1, 1]
	v = -(2.0*v - 1.0);
	u *= width / (float)height;
	u *= 2.0;
	v *= 2.0;
	vec3 eye = vec3(0, 0.5, 1.5);
	float distFrEye2Img = 1.0;;
	if ((x < width) && (y < height))
	{
		//for each pixel
		vec3 pixelPos = vec3(u, v, eye.z() - distFrEye2Img);
		//fire a ray:
		ray r;
		r.O = eye;
		r.Dir = pixelPos - eye; //view direction along negtive z-axis!
		vec3 col = castRay(r, d_world);
		float red = col.x();
		float green = col.y();
		float blue = col.z();
		d_output[i] = make_uchar4(red * 255, green * 255, blue * 255, 0);
	}
}


extern "C" void initTexture(int imageWidth, int imageHeight, uchar *h_data)
{
	// allocate array and copy image data
	cudaResourceDesc texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));
	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = d_imageArray;
}

extern "C" void freeTexture()
{
	checkCudaErrors(cudaFreeArray(d_imageArray));
}


// render image using CUDA
extern "C" void render(int width, int height, dim3 blockSize, dim3 gridSize,uchar4 *output, float *xPos, float *yPos, bool wind, bool gravity)
{	
	//make our world of hitables
	
	hitable **d_list;
	checkCudaErrors(cudaMalloc((void **)&d_list, ((numOfParticles + 5)) * sizeof(hitable *)));
	hitable **d_world;
	checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));

	float *device_xPos;
	float *device_yPos;

	checkCudaErrors(cudaMalloc((void**)&device_xPos, numOfParticles * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&device_yPos, numOfParticles * sizeof(float)));
	checkCudaErrors(cudaMemcpy(device_xPos, xPos, numOfParticles * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(device_yPos, yPos, numOfParticles * sizeof(float), cudaMemcpyHostToDevice));

	create_world << <1, 1 >> > (d_list, d_world);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	curandState* devStates;
	cudaMalloc(&devStates, 1 * sizeof(curandState));
	srand(time(0));

	int seed = rand();
	setup_kernel << <1, numOfParticles >> > (devStates, seed);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	if (gravity == true) {
		gravityKernel << <1, numOfParticles >> > (device_yPos);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	}
	
	
	if (wind == true) {
		windKernel << <1, numOfParticles>> > (device_xPos);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	}
	
	

	create_balls << <1, numOfParticles >> > (d_list, d_world, device_xPos, device_yPos,devStates);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	

	// call CUDA kernel, writing results to PBO memory
	d_render << <gridSize, blockSize >> > (output, width, height, d_world);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	checkCudaErrors(cudaMemcpy(xPos, device_xPos, numOfParticles * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(yPos, device_yPos, numOfParticles * sizeof(float), cudaMemcpyDeviceToHost));

	getLastCudaError("kernel failed");
}


#endif
