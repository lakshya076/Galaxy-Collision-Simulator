#include "cuda_engine.h"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#include <GL/gl.h>
#endif
#include <cuda_gl_interop.h>
#include <iostream>
#include <vector>
#include "morton.h"
#include "karras_bvh.cuh"

using namespace std;

// GPU SoA Arrays
float *d_x, *d_y, *d_z;
float *d_vx, *d_vy, *d_vz;
float *d_mass;
uint8_t *d_r, *d_g, *d_b;
uint8_t *d_type;
int *d_max_accel_int;

// Double Buffers for Sorting
float *d_x_alt, *d_y_alt, *d_z_alt;
float *d_vx_alt, *d_vy_alt, *d_vz_alt;
float *d_mass_alt;
uint8_t *d_r_alt, *d_g_alt, *d_b_alt;
uint8_t *d_type_alt;

float3* d_accels;

BvhNodeTraverse* d_bvh_nodes_tr;
BvhNodeBuild* d_bvh_nodes_bd;
uint64_t* d_morton_keys;
uint64_t* d_morton_keys_out;
int* d_indices;
int* d_indices_out;

void* d_temp_storage = nullptr;
size_t temp_storage_bytes = 0;

Star* d_staging_stars;
cudaGraphicsResource_t cuda_vbo_resource = nullptr;

template<typename T>
void swap_ptr(T*& a, T*& b) {
    T* temp = a;
    a = b;
    b = temp;
}

// Unpack initial Star structs into SoA arrays
__global__ void transpose_aos_to_soa_kernel(
    const Star* in_stars,
    float* x, float* y, float* z,
    float* vx, float* vy, float* vz,
    float* mass, uint8_t* r, uint8_t* g, uint8_t* b, uint8_t* type,
    int num_stars
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_stars) return;

    Star s = in_stars[i];
    x[i] = s.x; y[i] = s.y; z[i] = s.z;
    vx[i] = s.vx; vy[i] = s.vy; vz[i] = s.vz;
    mass[i] = s.mass;
    r[i] = s.r; g[i] = s.g; b[i] = s.b;
    type[i] = s.type;
}

// Pack SoA arrays back into Star structs for rendering or host copy
__global__ void transpose_soa_to_aos_kernel(
    Star* out_stars,
    const float* x, const float* y, const float* z,
    const float* vx, const float* vy, const float* vz,
    const float* mass, const uint8_t* r, const uint8_t* g, const uint8_t* b, const uint8_t* type,
    int num_stars
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_stars) return;

    out_stars[i].x = x[i]; out_stars[i].y = y[i]; out_stars[i].z = z[i];
    out_stars[i].vx = vx[i]; out_stars[i].vy = vy[i]; out_stars[i].vz = vz[i];
    out_stars[i].mass = mass[i];
    out_stars[i].r = r[i]; out_stars[i].g = g[i]; out_stars[i].b = b[i];
    out_stars[i].type = type[i];
}

// Gravity integration traversal using Binary BVH
__global__ void query_gravity_bvh_kernel(
    const float* x, const float* y, const float* z, const float* mass, 
    int num_stars, const BvhNodeTraverse* nodes, 
    float G, float epsilon_sq, float3* accels, int* max_accel_int
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_stars) return;

    float p_x = x[i];
    float p_y = y[i];
    float p_z = z[i];

    float ax = 0.0f, ay = 0.0f, az = 0.0f;

    // Fixed size stack
    int stack[256];
    int top = 0;
    
    stack[top++] = 0;

    while (top > 0) {
        int node_idx = stack[--top];
        
        float4 data0 = __ldg((const float4*)&nodes[node_idx]); 
        float4 data1 = __ldg(((const float4*)&nodes[node_idx]) + 1);

        float n_com_x = data0.x;
        float n_com_y = data0.y;
        float n_com_z = data0.z;
        float n_mass = data0.w;
        
        float width = data1.x;
        int child_left = __float_as_int(data1.y);
        int child_right = __float_as_int(data1.z);

        bool is_leaf = (node_idx >= num_stars - 1);

        float dx = n_com_x - p_x;
        float dy = n_com_y - p_y;
        float dz = n_com_z - p_z;
        float dist_sq = dx*dx + dy*dy + dz*dz + epsilon_sq;

        if (is_leaf) {
            if (dist_sq > epsilon_sq * 1.01f) {
                float inv_dist = rsqrtf(dist_sq);
                float inv_dist3 = inv_dist * inv_dist * inv_dist;
                float f = G * n_mass * inv_dist3;
                ax += dx * f;
                ay += dy * f;
                az += dz * f;
            }
        } else {
            if (width * width < dist_sq * 1.0f) {
                float inv_dist = rsqrtf(dist_sq);
                float inv_dist3 = inv_dist * inv_dist * inv_dist;
                float f = G * n_mass * inv_dist3;
                ax += dx * f;
                ay += dy * f;
                az += dz * f;
            } else {
                if (child_left != -1) stack[top++] = child_left;
                if (child_right != -1) stack[top++] = child_right;
            }
        }
    }
    
    accels[i] = make_float3(ax, ay, az);
    
    float accel_mag_sq = ax*ax + ay*ay + az*az;
    atomicMax(max_accel_int, __float_as_int(accel_mag_sq));
}

// Leapfrog integration
__global__ void integrate_kernel(
    float* x, float* y, float* z,
    float* vx, float* vy, float* vz,
    uint8_t* r, uint8_t* g, uint8_t* b,
    const uint8_t* type,
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
    
    // Dynamic Velocity-Based Coloring
    if (type[i] == 0) { // Galaxy A (Fire Gradient: Red -> Orange -> Yellow)
        float speed = sqrtf(vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i]);
        float t = fminf(speed / 150.0f, 1.0f);
        
        if (t < 0.5f) {
            float norm_t = t * 2.0f; // 0.0 to 1.0
            r[i] = 255;
            g[i] = (uint8_t)(50 + 150 * norm_t); // 50 to 200
            b[i] = 0;
        } else {
            float norm_t = (t - 0.5f) * 2.0f; // 0.0 to 1.0
            r[i] = 255;
            g[i] = (uint8_t)(200 + 55 * norm_t); // 200 to 255
            b[i] = (uint8_t)(150 * norm_t); // 0 to 150
        }
    } else if (type[i] == 2) { // Galaxy B (Ice Gradient: Dark Blue -> Cyan -> White)
        float speed = sqrtf(vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i]);
        float t = fminf(speed / 150.0f, 1.0f);
        
        if (t < 0.5f) {
            float norm_t = t * 2.0f; // 0.0 to 1.0
            r[i] = 0;
            g[i] = (uint8_t)(100 + 100 * norm_t); // 100 to 200
            b[i] = 255;
        } else {
            float norm_t = (t - 0.5f) * 2.0f; // 0.0 to 1.0
            r[i] = (uint8_t)(200 * norm_t); // 0 to 200
            g[i] = (uint8_t)(200 + 55 * norm_t); // 200 to 255
            b[i] = 255;
        }
    }
}

// Initialization
bool cuda_init(int num_stars) {
    cudaError_t err;

    int device_count = 0;
    err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        cerr << "CUDA Error: No CUDA-capable devices found." << endl;
        return false;
    }

    err = cudaSetDevice(0);
    if (err != cudaSuccess) return false;

    // Allocate primary SoA
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
    cudaMalloc(&d_type, num_stars * sizeof(uint8_t));
    
    cudaMalloc(&d_max_accel_int, sizeof(int));

    // Allocate alternate SoA for sorting double-buffering
    cudaMalloc(&d_x_alt, num_stars * sizeof(float));
    cudaMalloc(&d_y_alt, num_stars * sizeof(float));
    cudaMalloc(&d_z_alt, num_stars * sizeof(float));
    cudaMalloc(&d_vx_alt, num_stars * sizeof(float));
    cudaMalloc(&d_vy_alt, num_stars * sizeof(float));
    cudaMalloc(&d_vz_alt, num_stars * sizeof(float));
    cudaMalloc(&d_mass_alt, num_stars * sizeof(float));
    cudaMalloc(&d_r_alt, num_stars * sizeof(uint8_t));
    cudaMalloc(&d_g_alt, num_stars * sizeof(uint8_t));
    cudaMalloc(&d_b_alt, num_stars * sizeof(uint8_t));
    cudaMalloc(&d_type_alt, num_stars * sizeof(uint8_t));

    cudaMalloc(&d_accels, num_stars * sizeof(float3));

    // Allocate BVH arrays
    int num_nodes = 2 * num_stars - 1;
    cudaMalloc(&d_bvh_nodes_tr, num_nodes * sizeof(BvhNodeTraverse));
    cudaMalloc(&d_bvh_nodes_bd, num_nodes * sizeof(BvhNodeBuild));
    cudaMalloc(&d_morton_keys, num_stars * sizeof(uint64_t));
    cudaMalloc(&d_morton_keys_out, num_stars * sizeof(uint64_t));
    cudaMalloc(&d_indices, num_stars * sizeof(int));
    cudaMalloc(&d_indices_out, num_stars * sizeof(int));

    // Calculate CUB temp storage requirements
    cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, d_morton_keys, d_morton_keys_out, d_indices, d_indices_out, num_stars);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

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
    cudaFree(d_type); d_type = nullptr;
    cudaFree(d_max_accel_int); d_max_accel_int = nullptr;

    cudaFree(d_x_alt); d_x_alt = nullptr;
    cudaFree(d_y_alt); d_y_alt = nullptr;
    cudaFree(d_z_alt); d_z_alt = nullptr;
    cudaFree(d_vx_alt); d_vx_alt = nullptr;
    cudaFree(d_vy_alt); d_vy_alt = nullptr;
    cudaFree(d_vz_alt); d_vz_alt = nullptr;
    cudaFree(d_mass_alt); d_mass_alt = nullptr;
    cudaFree(d_r_alt); d_r_alt = nullptr;
    cudaFree(d_g_alt); d_g_alt = nullptr;
    cudaFree(d_b_alt); d_b_alt = nullptr;
    cudaFree(d_type_alt); d_type_alt = nullptr;

    cudaFree(d_accels); d_accels = nullptr;
    
    cudaFree(d_bvh_nodes_tr); d_bvh_nodes_tr = nullptr;
    cudaFree(d_bvh_nodes_bd); d_bvh_nodes_bd = nullptr;
    cudaFree(d_morton_keys); d_morton_keys = nullptr;
    cudaFree(d_morton_keys_out); d_morton_keys_out = nullptr;
    cudaFree(d_indices); d_indices = nullptr;
    cudaFree(d_indices_out); d_indices_out = nullptr;
    cudaFree(d_temp_storage); d_temp_storage = nullptr;

    cudaFree(d_staging_stars); d_staging_stars = nullptr;
}

void cuda_upload_initial_stars(const Star* host_stars, int num_stars) {
    cudaMemcpy(d_staging_stars, host_stars, num_stars * sizeof(Star), cudaMemcpyHostToDevice);
    int threads = 256;
    int blocks = (num_stars + threads - 1) / threads;
    transpose_aos_to_soa_kernel<<<blocks, threads>>>(d_staging_stars, 
                                                      d_x, d_y, d_z, 
                                                      d_vx, d_vy, d_vz, 
                                                      d_mass, d_r, d_g, d_b, 
                                                      d_type, num_stars);
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

// Executes one simulation frame natively on the GPU
float cuda_physics_step(
    Star* host_stars, 
    int num_stars, 
    float G, 
    float epsilon_sq, 
    float dt,
    float global_min,
    float global_max,
    bool use_interop
) {
    int threads = 256;
    int blocks_n = (num_stars + threads - 1) / threads;
    int blocks_n_minus_1 = (num_stars - 1 + threads - 1) / threads;

    // Reset max acceleration tracker
    cudaMemset(d_max_accel_int, 0, sizeof(int));

    compute_morton_and_indices_kernel<<<blocks_n, threads>>>(d_x, d_y, d_z, d_morton_keys, d_indices, num_stars, global_min, global_max);
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_morton_keys, d_morton_keys_out, d_indices, d_indices_out, num_stars);

    apply_sorted_indices_kernel<<<blocks_n, threads>>>(
        d_indices_out,
        d_x, d_y, d_z, d_vx, d_vy, d_vz, d_mass, d_r, d_g, d_b, d_type,
        d_x_alt, d_y_alt, d_z_alt, d_vx_alt, d_vy_alt, d_vz_alt, d_mass_alt, d_r_alt, d_g_alt, d_b_alt, d_type_alt,
        num_stars
    );

    // Swap pointers
    swap_ptr(d_x, d_x_alt); swap_ptr(d_y, d_y_alt); swap_ptr(d_z, d_z_alt);
    swap_ptr(d_vx, d_vx_alt); swap_ptr(d_vy, d_vy_alt); swap_ptr(d_vz, d_vz_alt);
    swap_ptr(d_mass, d_mass_alt); swap_ptr(d_r, d_r_alt); swap_ptr(d_g, d_g_alt); swap_ptr(d_b, d_b_alt);
    swap_ptr(d_type, d_type_alt);

    // Initialize BvhNodes
    init_internal_nodes_kernel<<<blocks_n_minus_1, threads>>>(num_stars, d_bvh_nodes_tr, d_bvh_nodes_bd);
    init_leaves_kernel<<<blocks_n, threads>>>(num_stars, d_bvh_nodes_tr, d_bvh_nodes_bd, d_x, d_y, d_z, d_mass);

    build_radix_tree_kernel<<<blocks_n_minus_1, threads>>>(num_stars, d_morton_keys_out, d_bvh_nodes_tr, d_bvh_nodes_bd);

    cudaDeviceSynchronize();
    aggregate_bvh_kernel<<<blocks_n, threads>>>(num_stars, d_bvh_nodes_tr, d_bvh_nodes_bd);

    cudaDeviceSynchronize();
    query_gravity_bvh_kernel<<<blocks_n, threads>>>(d_x, d_y, d_z, d_mass, num_stars, d_bvh_nodes_tr, G, epsilon_sq, d_accels, d_max_accel_int);

    integrate_kernel<<<blocks_n, threads>>>(d_x, d_y, d_z, d_vx, d_vy, d_vz, d_r, d_g, d_b, d_type, d_accels, dt, num_stars);

    // Render Mapping
    if (use_interop && cuda_vbo_resource) {
        Star* d_vbo_stars = nullptr;
        size_t num_bytes = 0;

        cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&d_vbo_stars, &num_bytes, cuda_vbo_resource);

        transpose_soa_to_aos_kernel<<<blocks_n, threads>>>(d_vbo_stars, 
                                                         d_x, d_y, d_z, 
                                                         d_vx, d_vy, d_vz, 
                                                         d_mass, d_r, d_g, d_b, 
                                                         d_type, num_stars);
        
        cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
    } else {
        transpose_soa_to_aos_kernel<<<blocks_n, threads>>>(d_staging_stars, 
                                                         d_x, d_y, d_z, 
                                                         d_vx, d_vy, d_vz, 
                                                         d_mass, d_r, d_g, d_b, 
                                                         d_type, num_stars);
        
        cudaMemcpy(host_stars, d_staging_stars, num_stars * sizeof(Star), cudaMemcpyDeviceToHost);
    }
    return dt;
}
