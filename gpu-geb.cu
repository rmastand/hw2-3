#include "common.h"
#include <cuda.h>
#include <iostream>

#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.

int blks; // number of GPU blocks
int NUM_BINS; // number of bins per side of the simulationa arena
int tot_num_bins; // total number of bins, with zero padding

// bins array:
// bins[k]=(index of first particle in bin k)
int *d_bins;

// particle linked list:
// part_links[i]=(index of next particle in the bin)
//              or (-1 if no more particles in the bin)
int *d_part_links;

__global__ void assign_particles_to_bins_gpu(int* bins, int* part_links, int num_parts, particle_t* parts, double size, int NUM_BLOCKS);
__global__ void initialize_array_gpu(int* array, int array_size);


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

    // get what bin the particle is in
    int dx = (particles[tid].x * NUM_BINS / size) + 1;
    int dy = (particles[tid].y * NUM_BINS / size) + 1;
    int bin_id = dx + (NUM_BINS+2)*dy;


    for (int m = -1; m <= 1; m++) {
        for (int n = -1; n <=1; n++) {

            ptcl_2_id = bins[bin_id + n + (NUM_BINS+2)*m];
            
            while (part_2_id >= 0) {
                apply_force(particles[tid], particles[ptcl_2_id]);
                ptcl_2_id = part_links[ptcl_2_id];
            }
        }
    }
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

    // NUM_THREADS is like `block_size`. In total we'll have
    // `num_parts` ptcls divided into blocks of size `NUM_THREADS`
    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;

    // call this "bins" (cells in the simulation grid) as distinguished
    // from "blocks" (groups of GPU threads)
    NUM_BINS = size/cutoff;
    tot_num_bins = (NUM_BINS+2)*(NUM_BINS+2);

    cudaMalloc((void **)&d_part_links, num_parts*sizeof(int));
    cudaMalloc((void **)&d_bins, (NUM_BINS+1)*sizeof(int));

}


void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // parts live in GPU memory
    
    /*
    TODO: 
    - Count ptcls per bin (in parallel) by iterating thru `parts`
    - Implement the array scheme described in lab?
    
    NOTE: `blks` as defined above means that each block of threads
    takes a fixed number of *particles*, not number of *bins*.
    */ 

    // Initialize the bins and particle links arrays
    initialize_array_gpu<<<blks, NUM_THREADS>>>(bins, tot_num_bins);
    initialize_array_gpu<<<blks, NUM_THREADS>>>(part_links, num_parts);

    assign_particles_to_bins_gpu<<<blks, NUM_THREADS>>>(bins, part_links,num_parts, parts, size, NUM_BLOCKS);

    // Compute forces
    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts);

    // Move particles
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
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

__global__ void assign_particles_to_bins_gpu(int* bins, int* part_links, int num_parts, particle_t* parts, double size, int NUM_BLOCKS) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Initialize the particle links array
     if (tid < num_parts) {
        // NOTE: don't need this bit? since we did initialize_array_gpu
        // atomicExch(&part_links[tid], -1);

        // Get particle's row and column, with padding
        int dx = (parts[tid].x * NUM_BLOCKS / size) + 1;
        int dy = (parts[tid].y * NUM_BLOCKS / size) + 1;
        int bin_id = dx + (NUM_BLOCKS+2)*dy;

        // Fill in the bins
        atomicExch(&part_links[tid], bins[bin_id]);
        atomicExch(&bins[bin_id], tid);
    }
}
