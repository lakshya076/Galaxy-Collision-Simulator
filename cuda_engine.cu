#include "cuda_engine.h"
#include <cuda_runtime.h>
#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#include <GL/gl.h>
#endif
#include <cuda_gl_interop.h>
#include <iostream>
#include <vector>

using namespace std;

// Maximum nodes based on the 64MB Arena size
const size_t MAX_NODES = (1024 * 1024 * 64) / sizeof(OctreeNode);
const size_t MAX_LEAF_INDICES = MAX_NODES * 32;

// GPU Device Pointers (SoA Simulation Space)
float* d_x = nullptr;
float* d_y = nullptr;
float* d_z = nullptr;
float* d_vx = nullptr;
float* d_vy = nullptr;
float* d_vz = nullptr;
float* d_mass = nullptr;
uint8_t* d_r = nullptr;
uint8_t* d_g = nullptr;
uint8_t* d_b = nullptr;
bool* d_is_dm = nullptr;
float3* d_accels = nullptr;

// Pre-allocated GPU Tree Structures
OctreeNode* d_nodes = nullptr;
float* d_node_masses = nullptr;
float* d_node_com_x = nullptr;
float* d_node_com_y = nullptr;
float* d_node_com_z = nullptr;
uint32_t* d_leaf_star_indices = nullptr;

// Staging buffer for non-interop Host to Device VBO transfers
Star* d_staging_stars = nullptr;

// CUDA GL Graphic resource for VBO interop
struct cudaGraphicsResource* cuda_vbo_resource = nullptr;

// Kernels

// Transpose AoS (Host layout) to SoA (GPU layout)
__global__ void transpose_aos_to_soa_kernel(const Star* aos_stars, 
                                            float* x, float* y, float* z,
                                            float* vx, float* vy, float* vz,
                                            float* mass, uint8_t* r, uint8_t* g, uint8_t* b,
                                            bool* is_dm, int num_stars) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_stars) return;

    Star s = aos_stars[i];
    x[i] = s.x;
    y[i] = s.y;
    z[i] = s.z;
    vx[i] = s.vx;
    vy[i] = s.vy;
    vz[i] = s.vz;
    mass[i] = s.mass;
    r[i] = s.r;
    g[i] = s.g;
    b[i] = s.b;
    is_dm[i] = s.is_dm;
}

// Transpose SoA (GPU layout) to AoS (Mapped VBO layout)
__global__ void transpose_soa_to_aos_kernel(Star* aos_stars,
                                            const float* x, const float* y, const float* z,
                                            const float* vx, const float* vy, const float* vz,
                                            const float* mass, const uint8_t* r, const uint8_t* g, const uint8_t* b,
                                            const bool* is_dm, int num_stars) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_stars) return;

    aos_stars[i].x = x[i];
    aos_stars[i].y = y[i];
    aos_stars[i].z = z[i];
    aos_stars[i].vx = vx[i];
    aos_stars[i].vy = vy[i];
    aos_stars[i].vz = vz[i];
    aos_stars[i].mass = mass[i];
    aos_stars[i].r = r[i];
    aos_stars[i].g = g[i];
    aos_stars[i].b = b[i];
    aos_stars[i].is_dm = is_dm[i];
}

// Gravitational solver using stack-based octree traversal
__global__ void query_gravity_kernel(
    const float* x, const float* y, const float* z,
    const float* mass,
    int num_stars,
    const OctreeNode* nodes,
    const OctreeNode* cpu_root_pointer,
    const float* node_masses,
    const float* node_com_x,
    const float* node_com_y,
    const float* node_com_z,
    const uint32_t* leaf_star_indices,
    float G,
    float epsilon_sq,
    float3* accels
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_stars) return;

    float my_x = x[i];
    float my_y = y[i];
    float my_z = z[i];

    float ax = 0.0f, ay = 0.0f, az = 0.0f;

    // Shared memory stack to avoid L1/L2 spills from divergent threads
    __shared__ int shared_stack[64 * 128]; // 64 depth max, 128 threads
    int stack_ptr = 0;
    shared_stack[stack_ptr * blockDim.x + threadIdx.x] = 0; // Root is at index 0
    stack_ptr++;

    while (stack_ptr > 0) {
        --stack_ptr;
        int node_idx = shared_stack[stack_ptr * blockDim.x + threadIdx.x];
        const OctreeNode& node = nodes[node_idx];

        if (node.is_leaf) {
            for (uint32_t j = 0; j < node.star_count; j++) {
                uint32_t star_idx = __ldg(&leaf_star_indices[node.start_star_index + j]);
                if (star_idx == i) continue;

                float dx = __ldg(&x[star_idx]) - my_x;
                float dy = __ldg(&y[star_idx]) - my_y;
                float dz = __ldg(&z[star_idx]) - my_z;

                float dist_sq = dx * dx + dy * dy + dz * dz + epsilon_sq;
                float inv_dist = rsqrtf(dist_sq);
                float inv_dist_cube = inv_dist * inv_dist * inv_dist;

                float f = G * __ldg(&mass[star_idx]) * inv_dist_cube;
                ax += dx * f;
                ay += dy * f;
                az += dz * f;
            }
        } else {
            // Calculate distance to center of mass
            float dx = __ldg(&node_com_x[node_idx]) - my_x;
            float dy = __ldg(&node_com_y[node_idx]) - my_y;
            float dz = __ldg(&node_com_z[node_idx]) - my_z;

            float dist_sq = dx * dx + dy * dy + dz * dz + epsilon_sq;

            // Barnes-Hut criteria (theta = 1.0 implicitly from original code)
            float width = node.half_width * 2.0f;
            if (width * width < dist_sq) {
                float inv_dist = rsqrtf(dist_sq);
                float inv_dist_cube = inv_dist * inv_dist * inv_dist;
                float f = G * __ldg(&node_masses[node_idx]) * inv_dist_cube;
                ax += dx * f;
                ay += dy * f;
                az += dz * f;
            } else {
                // Address child index on GPU using relative host pointer arithmetic
                int child_base_idx = ((const char*)node.first_child - (const char*)cpu_root_pointer) / sizeof(OctreeNode);
                uint8_t mask = node.active_mask;
                for (int c = 7; c >= 0; --c) {
                    if (mask & (1 << c)) {
                        if (stack_ptr < 64) {
                            shared_stack[stack_ptr * blockDim.x + threadIdx.x] = child_base_idx + c;
                            stack_ptr++;
                        }
                    }
                }
            }
        }
    }

    accels[i] = make_float3(ax, ay, az);
}

// Simple leapfrog integration step
__global__ void integrate_kernel(
    float* x, float* y, float* z,
    float* vx, float* vy, float* vz,
    const float3* accels,
    float dt,
    int num_stars
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_stars) return;

    vx[i] += accels[i].x * dt;
    vy[i] += accels[i].y * dt;
    vz[i] += accels[i].z * dt;

    x[i] += vx[i] * dt;
    y[i] += vy[i] * dt;
    z[i] += vz[i] * dt;
}

// Initialization
bool cuda_init(int num_stars) {
    cudaError_t err;

    // Check for GPU devices
    int device_count = 0;
    err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        cerr << "CUDA Error: No CUDA-capable devices found." << endl;
        return false;
    }

    err = cudaSetDevice(0);
    if (err != cudaSuccess) return false;

    // Allocate SoA Buffers
    cudaMalloc(&d_x, num_stars * sizeof(float));
    cudaMalloc(&d_y, num_stars * sizeof(float));
    cudaMalloc(&d_z, num_stars * sizeof(float));
    cudaMalloc(&d_vx, num_stars * sizeof(float));
    cudaMalloc(&d_vy, num_stars * sizeof(float));
    cudaMalloc(&d_vz, num_stars * sizeof(float));
    cudaMalloc(&d_mass, num_stars * sizeof(float));
    cudaMalloc(&d_r, num_stars * sizeof(uint8_t));
    cudaMalloc(&d_g, num_stars * sizeof(uint8_t));
    cudaMalloc(&d_b, num_stars * sizeof(uint8_t));
    cudaMalloc(&d_is_dm, num_stars * sizeof(bool));
    cudaMalloc(&d_accels, num_stars * sizeof(float3));

    // Pre-allocate large node arrays to avoid malloc overheads inside frames
    cudaMalloc(&d_nodes, MAX_NODES * sizeof(OctreeNode));
    cudaMalloc(&d_node_masses, MAX_NODES * sizeof(float));
    cudaMalloc(&d_node_com_x, MAX_NODES * sizeof(float));
    cudaMalloc(&d_node_com_y, MAX_NODES * sizeof(float));
    cudaMalloc(&d_node_com_z, MAX_NODES * sizeof(float));
    cudaMalloc(&d_leaf_star_indices, MAX_LEAF_INDICES * sizeof(uint32_t));

    // Staging array
    cudaMalloc(&d_staging_stars, num_stars * sizeof(Star));

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "CUDA Error during allocation: " << cudaGetErrorString(err) << endl;
        cuda_cleanup();
        return false;
    }

    return true;
}

void cuda_allocate_pinned_stars(Star** stars, int num_stars) {
    cudaMallocHost(stars, num_stars * sizeof(Star));
}

void cuda_free_pinned_stars(Star* stars) {
    cudaFreeHost(stars);
}

void cuda_allocate_pinned_playback(PlaybackStar** buf, int size) {
    cudaMallocHost(buf, size * sizeof(PlaybackStar));
}

void cuda_free_pinned_playback(PlaybackStar* buf) {
    cudaFreeHost(buf);
}

void cuda_cleanup() {
    cudaFree(d_x); d_x = nullptr;
    cudaFree(d_y); d_y = nullptr;
    cudaFree(d_z); d_z = nullptr;
    cudaFree(d_vx); d_vx = nullptr;
    cudaFree(d_vy); d_vy = nullptr;
    cudaFree(d_vz); d_vz = nullptr;
    cudaFree(d_mass); d_mass = nullptr;
    cudaFree(d_r); d_r = nullptr;
    cudaFree(d_g); d_g = nullptr;
    cudaFree(d_b); d_b = nullptr;
    cudaFree(d_is_dm); d_is_dm = nullptr;
    cudaFree(d_accels); d_accels = nullptr;

    cudaFree(d_nodes); d_nodes = nullptr;
    cudaFree(d_node_masses); d_node_masses = nullptr;
    cudaFree(d_node_com_x); d_node_com_x = nullptr;
    cudaFree(d_node_com_y); d_node_com_y = nullptr;
    cudaFree(d_node_com_z); d_node_com_z = nullptr;
    cudaFree(d_leaf_star_indices); d_leaf_star_indices = nullptr;
    
    cudaFree(d_staging_stars); d_staging_stars = nullptr;
}

void cuda_upload_initial_stars(const Star* host_stars, int num_stars) {
    // Stage upload
    cudaMemcpy(d_staging_stars, host_stars, num_stars * sizeof(Star), cudaMemcpyHostToDevice);

    // Unpack into SoA layout on GPU
    int threads = 256;
    int blocks = (num_stars + threads - 1) / threads;
    transpose_aos_to_soa_kernel<<<blocks, threads>>>(d_staging_stars, 
                                                      d_x, d_y, d_z, 
                                                      d_vx, d_vy, d_vz, 
                                                      d_mass, d_r, d_g, d_b, 
                                                      d_is_dm, num_stars);
    cudaDeviceSynchronize();
}

void cuda_register_vbo(unsigned int vbo_id) {
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo_id, cudaGraphicsRegisterFlagsNone);
}

void cuda_unregister_vbo() {
    if (cuda_vbo_resource) {
        cudaGraphicsUnregisterResource(cuda_vbo_resource);
        cuda_vbo_resource = nullptr;
    }
}

// Executes one simulation frame
void cuda_physics_step(
    Star* host_stars, 
    int num_stars, 
    float G, 
    float epsilon_sq, 
    float dt,
    const OctreeNode* nodes,
    const OctreeNode* cpu_root_pointer,
    int num_nodes,
    const float* node_masses,
    const float* node_com_x,
    const float* node_com_y,
    const float* node_com_z,
    const uint32_t* leaf_star_indices,
    bool use_interop
) {
    // Copy the tree metadata to GPU
    cudaMemcpy(d_nodes, nodes, num_nodes * sizeof(OctreeNode), cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_masses, node_masses, num_nodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_com_x, node_com_x, num_nodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_com_y, node_com_y, num_nodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_com_z, node_com_z, num_nodes * sizeof(float), cudaMemcpyHostToDevice);
    
    // Morton index arrays
    size_t active_leaf_indices = num_nodes * 32;
    cudaMemcpy(d_leaf_star_indices, leaf_star_indices, active_leaf_indices * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Launch gravity and integration kernels in parallel SoA space
    int threads = 128;
    int blocks = (num_stars + threads - 1) / threads;

    query_gravity_kernel<<<blocks, threads>>>(d_x, d_y, d_z, d_mass, num_stars, 
                                             d_nodes, cpu_root_pointer,
                                             d_node_masses, d_node_com_x, d_node_com_y, d_node_com_z, 
                                             d_leaf_star_indices, G, epsilon_sq, d_accels);
    
    integrate_kernel<<<blocks, threads>>>(d_x, d_y, d_z, d_vx, d_vy, d_vz, d_accels, dt, num_stars);

    // Render Mapping (VRAM to VRAM copy)
    if (use_interop && cuda_vbo_resource) {
        Star* d_vbo_stars = nullptr;
        size_t num_bytes = 0;

        // Map buffer for CUDA access
        cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&d_vbo_stars, &num_bytes, cuda_vbo_resource);

        // Pack the SoA coordinates directly inside the VBO
        transpose_soa_to_aos_kernel<<<blocks, threads>>>(d_vbo_stars, 
                                                         d_x, d_y, d_z, 
                                                         d_vx, d_vy, d_vz, 
                                                         d_mass, d_r, d_g, d_b, 
                                                         d_is_dm, num_stars);
        
        // Copy back to CPU stars array so host has position data for next frame's tree builder
        cudaMemcpy(host_stars, d_vbo_stars, num_stars * sizeof(Star), cudaMemcpyDeviceToHost);

        cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
    } else {
        // Fallback for Baking (or non-interop): Pack SoA into staging AoS block on GPU, and copy to host
        transpose_soa_to_aos_kernel<<<blocks, threads>>>(d_staging_stars, 
                                                         d_x, d_y, d_z, 
                                                         d_vx, d_vy, d_vz, 
                                                         d_mass, d_r, d_g, d_b, 
                                                         d_is_dm, num_stars);
        
        cudaMemcpy(host_stars, d_staging_stars, num_stars * sizeof(Star), cudaMemcpyDeviceToHost);
    }
}
