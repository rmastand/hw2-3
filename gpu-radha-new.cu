#include "common.h"
#include <cuda.h>
#include <iostream>
#include <thrust/scan.h>
#include <cuda/atomic>
#include <thrust/device_vector.h>


#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
int blks;

static int NUM_BLOCKS;
int tot_num_bins;

// Initialize arrays for particle ids and bin ids
int* bin_ids;
int* sorted_particles;
// Array to store how many particles in a bin have been added to sorted_particles
    // We use this to avoid conflicts when assigning each particle in index in the array sorted_particles
int* how_many_filled;

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

__global__ void compute_forces_gpu(particle_t* parts, int* sorted_particles, int* bin_ids, int num_parts, double size, int NUM_BLOCKS) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int loc_tid = tid; loc_tid < num_parts; loc_tid += stride ) {

        // Initialize acceleration to 0
        parts[loc_tid].ax = parts[loc_tid].ay = 0;

        // Get what row and column the particle would be in, with padding
        int dx = (parts[loc_tid].x * NUM_BLOCKS / size) + 1;
        int dy = (parts[loc_tid].y * NUM_BLOCKS / size) + 1;

        // Iterate through the 3x3 neighboring bins
        for (int m = -1; m <= 1; m++) {
            for (int n = -1; n <=1; n++) {

                // Get the bin_id of the neighboring bin
                int their_bin_id = dx + m + (NUM_BLOCKS+2)*(dy+n);

                // Iterate through all the particles in their_bin_id
                int their_bin_id_start = bin_ids[their_bin_id - 1];
                int next_bin_id_start = bin_ids[their_bin_id];

                for (int j = their_bin_id_start; j < next_bin_id_start; j++){

                    int particle_j_id = sorted_particles[j];
                    apply_force_gpu(parts[loc_tid], parts[particle_j_id]);
                }
            }
        }
    }
}

__global__ void move_gpu(particle_t* particles, int num_parts, double size) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int loc_tid = tid; loc_tid < num_parts; loc_tid += stride ) {

        particle_t* p = &particles[loc_tid];
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
}


__global__ void initialize_array_zeros_gpu(int* array, int array_size) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // Initialize the array with -1
    for (int loc_tid = tid; loc_tid < array_size; loc_tid += stride ) {
        array[loc_tid] = 0;
    }
 
}


__global__ void count_particles_per_bin(particle_t* parts, int* bin_ids, int num_parts, double size, int NUM_BLOCKS) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int loc_tid = tid; loc_tid < num_parts; loc_tid += stride ) {
        // Get what row and column the particle would be in, with padding
        int dx = (parts[loc_tid].x * NUM_BLOCKS / size) + 1;
        int dy = (parts[loc_tid].y * NUM_BLOCKS / size) + 1;
        // Get the bin id of the particle
        int bin_id = dx + (NUM_BLOCKS+2)*dy;
        
        // Increment the relevant bin_id
        atomicAdd(&bin_ids[bin_id], 1);
    }

}

__global__ void bin_particles(particle_t* parts, int* sorted_particles, int* bin_ids, int* how_many_filled, int num_parts, double size, int NUM_BLOCKS) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int loc_tid = tid; loc_tid < num_parts; loc_tid += stride ) {
        // Get what row and column the particle would be in, with padding
        int dx = (parts[loc_tid].x * NUM_BLOCKS / size) + 1;
        int dy = (parts[loc_tid].y * NUM_BLOCKS / size) + 1;
        // Get the bin id of the particle
        int bin_id = dx + (NUM_BLOCKS+2)*dy;

        // Get the id of where the particle will be stored in 
            // The particles for that bin start at position in array bin_ids[bin_id - 1] in sorted_particles
            // This particle goes to bin_ids[bin_id - 1] + loc_index
            // get loc_index from an atomic fetch_add in how_many_filled[bin_id]

        int bin_index_start = bin_ids[bin_id - 1]; 

        // Use some cuda atomics operation to fetch_add
            // Atomically read how_many_filled[bin_id] and at the same time increment it
        int loc_index =  atomicAdd(&how_many_filled[bin_id], 1);
        sorted_particles[bin_index_start + loc_index] = loc_tid; 
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

    cudaMalloc((void**)&bin_ids, tot_num_bins * sizeof(int));
    cudaMalloc((void**)&how_many_filled, tot_num_bins * sizeof(int));
    cudaMalloc((void**)&sorted_particles, num_parts * sizeof(int));
    
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // parts live in GPU memory

    // Initialize the array of bins_ids to have all 0's
    initialize_array_zeros_gpu<<<blks, NUM_THREADS>>>(bin_ids, tot_num_bins);
    // Initialize the array of how_many_filled to have all 0's
    initialize_array_zeros_gpu<<<blks, NUM_THREADS>>>(how_many_filled, tot_num_bins);
    // Initialize the array of how_many_filled to have all 0's
    initialize_array_zeros_gpu<<<blks, NUM_THREADS>>>(sorted_particles, num_parts);

    // Count the number of particles per bin
    count_particles_per_bin<<<blks, NUM_THREADS>>>(parts, bin_ids, num_parts, size, NUM_BLOCKS);

    // Parallel prefix sum
    thrust::inclusive_scan(thrust::device, bin_ids, bin_ids + tot_num_bins, bin_ids);

    // HORRIBLE NAMING but from this point, bin_ids represents bin_counts
    // The number of particles in bin_i is bin_counts[i] - bin_counts[i-1]
        // I have checked that this is the case
        // We don't need to worry about bin_id = 0 because that's a zero-pad bin

    // Add particles to an array ordered by what bin they're in
    bin_particles<<<blks, NUM_THREADS>>>(parts, sorted_particles, bin_ids, how_many_filled, num_parts, size, NUM_BLOCKS);

    // Compute forces
    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, sorted_particles, bin_ids, num_parts, size, NUM_BLOCKS);

    // Move particles
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
}
