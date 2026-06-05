#include <iostream>
#include <random>
#include <chrono>
#include <vector>
#include <omp.h>
#include "generator.h"
#include "engine.h"
#include "arena_allocator.h"
#include "gravity_aggregator.h"
#include <algorithm>
#include <parallel/algorithm>
#include "morton.h"

using namespace std;

struct Accel {
    float ax, ay, az;
};

int main() {
    int num_dm = 50000;
    int num_disc = 90000;
    int num_bulge = 10000;
    
    int particles_per_galaxy = num_dm + num_disc + num_bulge;
    int total_particles = particles_per_galaxy * 2;
    
    Star* stars = new Star[total_particles];
    int num_stars = total_particles;

    mt19937 global_rng(std::random_device{}());

    auto start_time = chrono::high_resolution_clock::now();
    
    cout << "Generating galaxies..." << endl;
    initial_setup(stars, num_dm, num_disc, num_bulge, global_rng);

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end_time - start_time;

    cout << "Collision scenario generated successfully!" << std::endl;
    cout << "Total stars spawned: " << total_particles << std::endl;
    cout << "Processing Time: " << diff.count() << " seconds" << endl;

    
    start_time = chrono::high_resolution_clock::now();
    float min_x = numeric_limits<float>::max(), min_y = numeric_limits<float>::max(), min_z = numeric_limits<float>::max();
    float max_x = numeric_limits<float>::lowest(), max_y = numeric_limits<float>::lowest(), max_z = numeric_limits<float>::lowest();

    for (size_t i = 0; i < num_stars; i++) {
        if (stars[i].x < min_x) min_x = stars[i].x;
        if (stars[i].y < min_y) min_y = stars[i].y;
        if (stars[i].z < min_z) min_z = stars[i].z;
        if (stars[i].x > max_x) max_x = stars[i].x;
        if (stars[i].y > max_y) max_y = stars[i].y;
        if (stars[i].z > max_z) max_z = stars[i].z;
    }

    float global_min = std::min({min_x, min_y, min_z});
    float global_max = std::max({max_x, max_y, max_z});

    cout << "Sorting stars using Morton Z-Order Curve..." << endl;
    __gnu_parallel::sort(stars, stars + num_stars, [global_min, global_max](const Star& a, const Star& b) {
        return get_morton_code(a.x, a.y, a.z, global_min, global_max) < 
               get_morton_code(b.x, b.y, b.z, global_min, global_max);
    });

    ArenaAllocator arena(1024 * 1024 * 64);
    OctreeNode* root = arena.alloc<OctreeNode>();

    float max_span = std::max({max_x - min_x, max_y - min_y, max_z - min_z});
    root->half_width = max_span * 0.5f;
    root->center_x = (min_x + max_x) * 0.5f;
    root->center_y = (min_y + max_y) * 0.5f;
    root->center_z = (min_z + max_z) * 0.5f;
    root->is_leaf = true;
    root->star_count = 0;

    const size_t max_nodes = (1024 * 1024 * 64) / sizeof(OctreeNode);
    uint32_t* leaf_star_indices = new uint32_t[max_nodes * 32];
    uint32_t next_free_idx = 0;
    root->start_star_index = next_free_idx;
    next_free_idx += 32;

    cout << "\nBuilding Basic Octree:" << endl;
    for (uint32_t i = 0; i < num_stars; i++) {
        insert_star(root, i, stars, arena, leaf_star_indices, next_free_idx);
    }

    end_time = chrono::high_resolution_clock::now();
    diff = end_time - start_time;

    cout << "\nTotal Initialization Time: " << diff.count() << " seconds" << endl;
    cout << "Arena Memory Consumed: " << (arena.get_used_memory() / 1024.0 / 1024.0) << " MB" << endl;

    float* node_masses = new float[max_nodes];
    float* node_com_x = new float[max_nodes];
    float* node_com_y = new float[max_nodes];
    float* node_com_z = new float[max_nodes];

    cout << "\nPopulating Node Metadata" << endl;
    node_physics(root, root, stars, node_masses, node_com_x, node_com_y, node_com_z, leaf_star_indices);

    cout << "\nRunning Gravity Query and Integration step for all stars..." << endl;
    start_time = chrono::high_resolution_clock::now();

    float G = 1.0f;
    float epsilon_sq = 1e-3f; // Softening parameter
    float dt = 0.01f;

    std::vector<Accel> accels(num_stars);

    #ifdef _OPENMP
    cout << "Using " << omp_get_max_threads() << " OpenMP threads for calculation." << endl;
    #else
    cout << "Using 1 thread (OpenMP disabled)." << endl;
    #endif

    #pragma omp parallel for schedule(dynamic, 256)
    for (int i = 0; i < num_stars; ++i) {
        query_gravity(
            root, stars[i], i, stars, 
            node_masses, node_com_x, node_com_y, node_com_z, leaf_star_indices,
            G, epsilon_sq, accels[i].ax, accels[i].ay, accels[i].az
        );
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_stars; ++i) {
        // Update velocity
        stars[i].vx += accels[i].ax * dt;
        stars[i].vy += accels[i].ay * dt;
        stars[i].vz += accels[i].az * dt;

        // Update position
        stars[i].x += stars[i].vx * dt;
        stars[i].y += stars[i].vy * dt;
        stars[i].z += stars[i].vz * dt;
    }

    end_time = chrono::high_resolution_clock::now();
    diff = end_time - start_time;
    cout << "Gravity Calculation & Step Time: " << diff.count() << " seconds" << endl;

    delete[] leaf_star_indices;
    delete[] node_masses;
    delete[] node_com_x;
    delete[] node_com_y;
    delete[] node_com_z;
    delete[] stars;

    return 0;
}