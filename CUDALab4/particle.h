#pragma once
#include "cuda_runtime.h"
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include "time.h"

class Particle {
public:

	float x;
	float y;
	Particle();
	__device__ __host__ Particle(float x, float y) : x(x), y(y) {};

};


class ParticleSystem {
public:
	ParticleSystem(int numOfParticles);
	std::vector<Particle> particleList = std::vector<Particle>();
	void initialise(int numOfParticles);
	__device__ __host__ float getParticleX(int index);
	void printAllParticles();
	int numberOfParticles; 

};

ParticleSystem::ParticleSystem(int numOfParticles) {
	initialise(numOfParticles);
}

void ParticleSystem::initialise(int numOfParticles) {
	srand(time(0));
	for (int i = 0; i < numOfParticles; i++) {
		float xPos = (4.0 * ((rand() % 100) * 0.01)) - 2.0;
		float yPos = (4.0 * ((rand() % 100) * 0.01)) - 2.0;
		Particle p = Particle(xPos, yPos);
		particleList.push_back(p);
	}
}

void ParticleSystem::printAllParticles() {
	for (Particle p : particleList) {
		std::cout << "X: " << p.x << " Y: " << p.y << std::endl;
	}
}


__device__ __host__ float ParticleSystem::getParticleX(int index) {
	return particleList[index].x;
}


