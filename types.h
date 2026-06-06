#pragma once

#include <cstdint>

struct Star {
    float x, y, z;    
    float vx, vy, vz;
    float mass;

    bool is_dm;
    uint8_t r, g, b;
}; 

struct PlaybackStar {
    float x, y, z;
    uint8_t r, g, b;
    uint8_t padding;
}; // used to move the Star metadata in simulation.bin file for OpenGL and saving space

// 32-byte strictly aligned
struct alignas(32) OctreeNode {
    float center_x, center_y, center_z; // 12 bytes
    float half_width;                   // 4 bytes

    union {
        OctreeNode* first_child; // 8 bytes
        struct {
            uint32_t start_star_index; // 4 bytes
            uint32_t star_count;       // 4 bytes
        };
    }; // 8 bytes

    bool is_leaf; // 1 byte
    uint8_t active_mask; // 1 byte
    uint8_t padding[6]; // 6 bytes padding to reach 32 bytes
};

// Compile-time assertion to guarantee half-L1 Cache-Line layout
static_assert(sizeof(OctreeNode) == 32, "OctreeNode must be exactly 32 bytes for optimized cache alignment");
