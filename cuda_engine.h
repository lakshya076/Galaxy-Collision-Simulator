#pragma once
#include "types.h"


bool cuda_init(int num_stars);
void cuda_cleanup();

// Allocation of page-locked host memory
void cuda_allocate_pinned_stars(Star** stars, int num_stars);
void cuda_free_pinned_stars(Star* stars);
void cuda_allocate_pinned_playback(PlaybackStar** buf, int size);
void cuda_free_pinned_playback(PlaybackStar* buf);

// Copy initial conditions into GPU SoA arrays
void cuda_upload_initial_stars(const Star* host_stars, int num_stars);

// OpenGL GL-CUDA Resource Registration
void cuda_register_vbo(unsigned int vbo_id);
void cuda_unregister_vbo();

// Physics Step
float cuda_physics_step(
    Star* host_stars, 
    int num_stars, 
    float G, 
    float epsilon_sq, 
    float dt,
    float global_min, 
    float global_max, 
    bool use_interop
);
