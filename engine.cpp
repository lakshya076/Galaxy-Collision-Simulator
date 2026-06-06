#include "engine.h"

using namespace std;

void set_child_bounds(OctreeNode* parent, OctreeNode* child, int index) {
    float quarter = parent->half_width * 0.5f;
    child->half_width = quarter;
    child->center_x = parent->center_x + ((index & 1) ? quarter : -quarter);
    child->center_y = parent->center_y + ((index & 2) ? quarter : -quarter);
    child->center_z = parent->center_z + ((index & 4) ? quarter : -quarter);

    child->is_leaf = true;
    child->star_count = 0;
}

int get_child_index(OctreeNode* node, float x, float y, float z) {
    int idx = 0;
    idx |= (x >= node->center_x) ? 1 : 0;
    idx |= (y >= node->center_y) ? 2 : 0;
    idx |= (z >= node->center_z) ? 4 : 0;
    return idx;
}

uint32_t get_node_index(const OctreeNode* root, const OctreeNode* node) {
    return static_cast<uint32_t>(node - root);
}

void insert_star(OctreeNode* node, uint32_t star_idx, const Star* all_stars, ArenaAllocator& arena, uint32_t* leaf_indices, uint32_t& next_free_idx) {
    if (node->is_leaf) {
        if (node->star_count < 32) {
            leaf_indices[node->start_star_index + node->star_count] = star_idx;
            node->star_count++;
        } else {
            node->is_leaf = false;
            
            uint32_t old_count = node->star_count;
            uint32_t old_start = node->start_star_index;

            // Allocate 256 bytes for 8 contiguous children (32 bytes each)
            OctreeNode* children = arena.alloc_array<OctreeNode>(8);
            node->first_child = children;

            for (int i = 0; i < 8; i++) {
                set_child_bounds(node, &children[i], i);
                children[i].start_star_index = next_free_idx;
                next_free_idx += 32;
            }

            // Route the original stars into new children
            for (uint32_t i = 0; i < old_count; i++) {
                uint32_t s_idx = leaf_indices[old_start + i];
                int c_idx = get_child_index(node, all_stars[s_idx].x, all_stars[s_idx].y, all_stars[s_idx].z);
                insert_star(&children[c_idx], s_idx, all_stars, arena, leaf_indices, next_free_idx);
            }

            // Route the new star
            int c_idx = get_child_index(node, all_stars[star_idx].x, all_stars[star_idx].y, all_stars[star_idx].z);
            insert_star(&children[c_idx], star_idx, all_stars, arena, leaf_indices, next_free_idx);
        }
    } else {
        int c_idx = get_child_index(node, all_stars[star_idx].x, all_stars[star_idx].y, all_stars[star_idx].z);
        insert_star(&node->first_child[c_idx], star_idx, all_stars, arena, leaf_indices, next_free_idx);
    }
}


void node_physics(OctreeNode* node, OctreeNode* root, const Star* stars, float* node_masses, 
    float* node_com_x, float* node_com_y, float* node_com_z, const uint32_t* leaf_indices) {
        uint32_t idx = get_node_index(root, node);

        if (node->is_leaf) {
            float total_mass = 0.0f;
            float weighted_x = 0.0f, weighted_y = 0.0f, weighted_z = 0.0f;

            for (uint32_t i = 0; i < node->star_count; i++) {
                uint32_t star_idx = leaf_indices[node->start_star_index + i];
                float curr_mass = stars[star_idx].mass;
                total_mass += curr_mass;
                weighted_x += stars[star_idx].x * curr_mass;
                weighted_y += stars[star_idx].y * curr_mass;
                weighted_z += stars[star_idx].z * curr_mass;
            }

            node_masses[idx] = total_mass;

            if(total_mass > 0.0f) {
                node_com_x[idx] = weighted_x / total_mass;
                node_com_y[idx] = weighted_y / total_mass;
                node_com_z[idx] = weighted_z / total_mass;
            } else {
                node_com_x[idx] = node->center_x;
                node_com_y[idx] = node->center_y;
                node_com_z[idx] = node->center_z;
            }
        } else {
            float total_mass = 0.0f;
            float weighted_x = 0.0f, weighted_y = 0.0f, weighted_z = 0.0f;
            uint8_t mask = 0;
            
            for (int i = 0; i < 8; ++i) {
                OctreeNode* child = &node->first_child[i];
                node_physics(child, root, stars, node_masses, node_com_x, node_com_y, node_com_z, leaf_indices);

                uint32_t child_idx = get_node_index(root, child);
                float cm = node_masses[child_idx];
                if (cm > 0.0f) {
                    mask |= (1 << i);
                }
                total_mass += cm;
                weighted_x += node_com_x[child_idx] * cm;
                weighted_y += node_com_y[child_idx] * cm;
                weighted_z += node_com_z[child_idx] * cm;
            }

            node->active_mask = mask;
            node_masses[idx] = total_mass;
            if (total_mass > 0.0f) {
                node_com_x[idx] = weighted_x / total_mass;
                node_com_y[idx] = weighted_y / total_mass;
                node_com_z[idx] = weighted_z / total_mass;
            } else {
                node_com_x[idx] = node->center_x;
                node_com_y[idx] = node->center_y;
                node_com_z[idx] = node->center_z;
            }
        }
}