#include "common.h"
#include <cuda.h>
#include <iostream>

#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
int blks;
static int NUM_BLOCKS;
int tot_num_bins;
int* bins;
int* part_links;


__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    // r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    //
    //  very simple short-range repulsive force
    //
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

__global__ void compute_forces_gpu(particle_t* particles, int num_parts) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particles[tid].ax = particles[tid].ay = 0;
    for (int j = 0; j < num_parts; j++)
        apply_force_gpu(particles[tid], particles[j]);
}

__global__ void move_gpu(particle_t* particles, int num_parts, double size) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particle_t* p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    //
    //  bounce from walls
    //
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
        p->vy = -(p->vy);
    }
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here

    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    

    // num blocks in either x or y direction (+2 in each dimension for padding)
    NUM_BLOCKS = size/cutoff;
    tot_num_bins = (NUM_BLOCKS+2)*(NUM_BLOCKS+2);

    cudaMalloc((void**)&bins, tot_num_bins * sizeof(int));
    cudaMalloc((void**)&part_links, num_parts * sizeof(int));

}

__global__ void initialize_array_gpu(int* array, int array_size) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Initialize the array with -1
    if (tid >= array_size) {
        return; 
    }
        array[tid] = -1;
    }

/*
__global__ void assign_particles_to_bins_gpu(int* bins, int* part_links, int num_parts, particle_t* parts) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Initialize the particle links array
     if (tid < num_parts) {
        atomicExch(&part_links[tid], -1);

        // Get what row and column the particle would be in, with padding
        int dx = (parts[tid].x * NUM_BLOCKS / size) + 1;
        int dy = (parts[tid].y * NUM_BLOCKS / size) + 1;
        int bin_id = dx + (NUM_BLOCKS+2)*dy;

        // Fill in the bins
        atomicExch(&part_links[tid], bins[bin_id]);
        atomicExch(& bins[bin_id], tid);
    }

}
*/

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // parts live in GPU memory
    // Rewrite this function

    // Initialize the bins array
    initialize_array_gpu<<<blks, NUM_THREADS>>>(bins, tot_num_bins);
    // Initialize the particle links array
    initialize_array_gpu<<<blks, NUM_THREADS>>>(part_links, num_parts);

    // test
    int* bins_cpu = (int*) malloc(tot_num_bins * sizeof(int));
    int* part_links_cpu = (int*) malloc(num_parts * sizeof(int));

    cudaMemcpy(bins_cpu, bins, tot_num_bins * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(part_links_cpu, part_links, num_parts * sizeof(int), cudaMemcpyDeviceToHost);

    for (int p = 0; p < tot_num_bins; p++) {
            std::cout << "testing bins " << p << " " << bins_cpu[p] << std::endl;
    }
    for (int p = 0; p < num_parts; p++) {
            std::cout << "testing links " << p << " " << part_links_cpu[p] << std::endl;
    }
  
    
    std::cout << "b" << std::endl;

    //assign_particles_to_bins_gpu<<<blks, NUM_THREADS>>>(bins, part_links,num_parts, parts)


  
    std::cout << "c" << std::endl;

    // Compute forces
    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts);

    // Move particles
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
}
