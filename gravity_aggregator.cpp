#include "gravity_aggregator.h"

void query_gravity(
    const OctreeNode* root,
    const Star& star,
    uint32_t target_idx,
    const Star* all_stars,
    const float* node_masses,
    const float* node_com_x,
    const float* node_com_y,
    const float* node_com_z,
    const uint32_t* leaf_indices,
    float G,
    float epsilon_sq,
    float& ax, float& ay, float& az
) {
    ax = 0.0f; ay = 0.0f; az = 0.0f;
    
    // Explicit stack to replace recursion.
    const OctreeNode* stack[128];
    int stack_ptr = 0;
    stack[stack_ptr++] = root;
    
    float theta_sq = 0.8f * 0.8f;
    
    while (stack_ptr > 0) {
        const OctreeNode* node = stack[--stack_ptr];
        
        uint32_t node_idx = get_node_index(root, node);
        if (node_masses[node_idx] == 0.0f) continue;

        float s = node->half_width * 2.0f;
        float dx = node_com_x[node_idx] - star.x;
        float dy = node_com_y[node_idx] - star.y;
        float dz = node_com_z[node_idx] - star.z;
        float dist_sq = dx * dx + dy * dy + dz * dz;

        if (node->is_leaf || (s * s < theta_sq * dist_sq)) {
            if (node->is_leaf) {
                // Particle-to-Particle (Direct Gravity) inside leaf node
                for (uint32_t i = 0; i < node->star_count; i++) {
                    if (i + 1 < node->star_count) {
                        uint32_t next_s_idx = leaf_indices[node->start_star_index + i + 1];
                        __builtin_prefetch(&all_stars[next_s_idx], 0, 1);
                    }

                    uint32_t s_idx = leaf_indices[node->start_star_index + i];
                    if (s_idx == target_idx) continue;
                    float sdx = all_stars[s_idx].x - star.x;
                    float sdy = all_stars[s_idx].y - star.y;
                    float sdz = all_stars[s_idx].z - star.z;
                    
                    float r_sq = sdx * sdx + sdy * sdy + sdz * sdz + epsilon_sq;
                    float inv_r = 1.0f / std::sqrt(r_sq); 
                    float inv_r3 = inv_r * inv_r * inv_r;
                    float acc_mag = G * all_stars[s_idx].mass * inv_r3;
                    ax += acc_mag * sdx;
                    ay += acc_mag * sdy;
                    az += acc_mag * sdz;
                }
            } else {
                // Approximation: Treat the whole node as a single macro-particle
                float r_sq = dist_sq + epsilon_sq;
                float inv_r = 1.0f / std::sqrt(r_sq);
                float inv_r3 = inv_r * inv_r * inv_r;
                float acc_mag = G * node_masses[node_idx] * inv_r3;
                ax += acc_mag * dx;
                ay += acc_mag * dy;
                az += acc_mag * dz;
            }
        } else {
            // Node is too close; push the 8 children onto the stack
            for (int i = 0; i < 8; i++) {
                stack[stack_ptr++] = &node->first_child[i];
            }
        }
    }
}