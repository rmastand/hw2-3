#include "common.h"
#include <cuda.h>

#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
int blks;
__global__ static int NUM_BLOCKS;
__global__ static int tot_num_bins;
__global__ int* bins_gpu;
__global__ int* part_links_gpu;


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

    bins = (int*) malloc(tot_num_bins * sizeof(int));
    part_links = (int*) malloc(num_parts * sizeof(int));

    int* bins_gpu;
    int* part_links_gpu;

    cudaMalloc((void**)&bins, tot_num_bins * sizeof(int));
    cudaMemcpy(bins_gpu, bins, tot_num_bins * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&part_links, num_parts * sizeof(int));
    cudaMemcpy(part_links_gpu, part_links, num_parts * sizeof(int), cudaMemcpyHostToDevice);

}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // parts live in GPU memory
    // Rewrite this function

    // Fill the bins array
    for (int i = 0; i < tot_num_bins; i++) {
			bins_gpu[i] = -1;
		}

    // Fill the particle links array
    for (int i = 0; i < num_parts; i++) {
        part_links_gpu[i] = -1;
    }

    // Assign particles to bins
    for (int i = 0; i < num_parts; ++i) {
        // Get what row and column the particle would be in, with padding
        int dx = (parts[i].x * NUM_BLOCKS / size) + 1;
        int dy = (parts[i].y * NUM_BLOCKS / size) + 1;
        int bin_id = dx + (NUM_BLOCKS+2)*dy;

        // -1 if bin holds -1, the bin's particle id otherwise
        part_links_gpu[i] = bins_gpu[bin_id];
        bins_gpu[bin_id] = i;
    }

    // Now compute forces between part_1 and part_2
    int part_1_id;
	int part_2_id;

	// Defined globally
    // THIS I NEED TO FIGURE OUT -- HOW TO WE MAP THIS 2D THREADING TO THE BLOCKS AND THREADS OF THE GPU??
    // I think I want a 2d block structure -- will need to ask this in OH
	int Nthrds, Nxthrds, Nythrds, delX, delY;	
	Nxthrds = sqrt(Nthrds); // number of divisions in x
	Nythrds = Nthrds / Nxthrds; // number of divisions in y
	delX = (NUM_BLOCKS / Nxthrds) + 1; // side of division in x
	delY = (NUM_BLOCKS / Nythrds) + 1; // side of division in x

    int id, threadX, threadY;

	// Defined only for a thread
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	threadX = id % Nxthrds; 
	threadY = id / Nxthrds;


	for (int ddx = delX*threadX; ddx < min(delX*(threadX + 1), NUM_BLOCKS); ddx++) {
			for (int ddy = delY*threadY; ddy < min(delY*(threadY+1), NUM_BLOCKS); ddy++) {


			int bin_id_pad = (ddx+1) + (NUM_BLOCKS+2)*(ddy+1);
			part_1_id = bins[bin_id_pad];


			//concept here: 
			//use part_1_id to track current particle of interest in bin[bin_id_pad]
			//	skip bin if empty (because id=0), otherwise traverse its linked list until end (when id=-1)
			//	look at particle[part_1_id] 
			//	compute forces with all particles in neighboring 9 bins, by traversing THEIR linked lists with part_2_id

			while (part_1_id >= 0) {

				//cout << "begin " << part_1_id << " " << id << " end\n" << endl;

				for (int m = -1; m <= 1; m++) {
					for (int n = -1; n <=1; n++) {
						part_2_id = bins[bin_id_pad + n + (NUM_BLOCKS+2)*m];
						while (part_2_id >= 0) {
							apply_force(parts[part_1_id], parts[part_2_id]);
							
							part_2_id = part_links[part_2_id];
						}
					}
				}
				part_1_id = part_links[part_1_id];
			}
		}
	}
	
	
    for (int i = 0; i < num_parts; ++i) {
        move(parts[i], size);
    }
	




    // Compute forces
    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts);

    // Move particles
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
}
