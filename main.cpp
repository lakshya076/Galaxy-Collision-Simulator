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
#if !defined(_MSC_VER)
#include <parallel/algorithm>
#endif
#include "morton.h"
#include "renderer.h"
#include <cstdio>
#include <string>

#include <thread>
#include "cuda_engine.h"

using namespace std;

struct Accel {
    float ax, ay, az;
};

int main(int argc, char** argv) {
    int num_dm = 50000;
    int num_disc = 90000;
    int num_bulge = 10000;
    
    int particles_per_galaxy = num_dm + num_disc + num_bulge;
    int total_particles = particles_per_galaxy * 2;
    
    bool use_gpu = true;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--cpu") {
            use_gpu = false;
        }
    }

    if (use_gpu) {
        if (!cuda_init(total_particles)) {
            std::cout << "Falling back to CPU OpenMP mode..." << std::endl;
            use_gpu = false;
        } else {
            std::cout << "GPU Acceleration initialized successfully." << std::endl;
        }
    }

    Star* stars = nullptr;
    #ifndef MODE_PLAYBACK
    if (use_gpu) {
        cuda_allocate_pinned_stars(&stars, total_particles);
    } else {
        stars = new Star[total_particles];
    }
    #else
    stars = new Star[total_particles];
    #endif

    int num_stars = total_particles;
    int num_visible = 2 * (num_disc + num_bulge);

    mt19937 global_rng(std::random_device{}());

    auto start_time = chrono::high_resolution_clock::now();
    
    cout << "Generating galaxies..." << endl;
    initial_setup(stars, num_dm, num_disc, num_bulge, global_rng);

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end_time - start_time;

    cout << "Collision scenario generated successfully in " << diff.count() << "s" << std::endl;
    cout << "Total stars spawned: " << total_particles << std::endl;

    if (use_gpu) {
        cuda_upload_initial_stars(stars, num_stars);
    }
    
#ifndef MODE_BAKE
    // Initialize OpenGL for Live or Playback mode
    GLFWwindow* window = init_opengl(1920, 1080);
    if (!window) {
        return -1;
    }
    #if defined(MODE_PLAYBACK)
        setup_buffers(stars, num_stars);
    #else
        setup_buffers(stars, num_stars);
        if (use_gpu) {
            cuda_register_vbo(get_vbo_handle());
        }
    #endif
#else
    GLFWwindow* window = nullptr;
#endif

    // Hoist large allocations
    ArenaAllocator arena(1024 * 1024 * 64);
    const size_t max_nodes = (1024 * 1024 * 64) / sizeof(OctreeNode);
    uint32_t* leaf_star_indices = new uint32_t[max_nodes * 32];
    float* node_masses = new float[max_nodes];
    float* node_com_x = new float[max_nodes];
    float* node_com_y = new float[max_nodes];
    float* node_com_z = new float[max_nodes];
    std::vector<Accel> accels(num_stars);

    float G = 1.0f;
    float epsilon_sq = 10.0f;
    double simulation_time = 0.0;

    cout << "\nStarting Simulation..." << endl;
    
    int frame_count = 0;

#if defined(MODE_BAKE)
    cout << "Writing raw simulation data to simulation.bin..." << endl;
    FILE* out_file = fopen("simulation.bin", "wb");
    if (!out_file) {
        cerr << "Failed to open simulation.bin for writing" << endl;
        return -1;
    }

    int total_frames = 1000;
    if (argc > 1) {
        total_frames = std::stoi(argv[1]);
        cout << "Baking " << total_frames << " frames." << endl;
    }

    // Allocate Pinned Host Buffers for double-buffered SSD writes
    PlaybackStar* host_buf_A = nullptr;
    PlaybackStar* host_buf_B = nullptr;
    if (use_gpu) {
        cuda_allocate_pinned_playback(&host_buf_A, num_stars);
        cuda_allocate_pinned_playback(&host_buf_B, num_stars);
    } else {
        host_buf_A = new PlaybackStar[num_stars];
        host_buf_B = new PlaybackStar[num_stars];
    }

    std::thread writer_thread;
    PlaybackStar* current_write_buf = host_buf_A;
    PlaybackStar* next_write_buf = host_buf_B;

    double total_time = 0;
    for (frame_count = 0; frame_count < total_frames; ++frame_count) {
        start_time = chrono::high_resolution_clock::now();

        // Calculate Bounds
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

        auto t_sort_start = chrono::high_resolution_clock::now();

        // Morton Sort Optimization: Precompute keys
        std::vector<std::pair<uint64_t, Star>> sort_buffer(num_stars);
        
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < num_stars; i++) {
            sort_buffer[i].first = get_morton_code(stars[i].x, stars[i].y, stars[i].z, global_min, global_max);
            sort_buffer[i].second = stars[i];
        }

#if defined(_MSC_VER)
        std::sort(sort_buffer.begin(), sort_buffer.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
#else
        __gnu_parallel::sort(sort_buffer.begin(), sort_buffer.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
#endif

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < num_stars; i++) {
            stars[i] = sort_buffer[i].second;
        }


        if (use_gpu) {
            float actual_dt = cuda_physics_step(
                stars, num_stars, G, epsilon_sq, 0.01f,
                global_min, global_max, false
            );
            simulation_time += actual_dt;
        } else {
            // CPU fallback logic
            float dt = 0.01f;
            simulation_time += dt;
            auto t_build_start = chrono::high_resolution_clock::now();

            // Reset Arena and Build CPU Octree
            arena.reset();
            OctreeNode* root = arena.alloc<OctreeNode>();

            float max_span = std::max({max_x - min_x, max_y - min_y, max_z - min_z});
            root->center_x = (min_x + max_x) * 0.5f;
            root->center_y = (min_y + max_y) * 0.5f;
            root->center_z = (min_z + max_z) * 0.5f;
            root->half_width = max_span * 0.5f;
            root->is_leaf = true;
            root->start_star_index = 0;
            root->star_count = 0;

            uint32_t next_free_idx = 0;
            root->start_star_index = next_free_idx;
            next_free_idx += 32;

            for (uint32_t i = 0; i < num_stars; i++) {
                insert_star(root, i, stars, arena, leaf_star_indices, next_free_idx);
            }

            int num_nodes = arena.get_used_memory() / sizeof(OctreeNode);
            
            node_physics(root, root, stars, node_masses, node_com_x, node_com_y, node_com_z, leaf_star_indices);

            // Gravity Queries
            #pragma omp parallel for schedule(dynamic, 256)
            for (int i = 0; i < num_stars; ++i) {
                query_gravity(
                    root, stars[i], i, stars, 
                    node_masses, node_com_x, node_com_y, node_com_z, leaf_star_indices,
                    G, epsilon_sq, accels[i].ax, accels[i].ay, accels[i].az
                );
            }

            // Velocity / Position Integration
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < num_stars; ++i) {
                stars[i].vx += accels[i].ax * dt;
                stars[i].vy += accels[i].ay * dt;
                stars[i].vz += accels[i].az * dt;

                stars[i].x += stars[i].vx * dt;
                stars[i].y += stars[i].vy * dt;
                stars[i].z += stars[i].vz * dt;
            }
        }

        auto t_gpu_end = chrono::high_resolution_clock::now();

        // Wait for the background writer thread from the previous frame to finish
        if (writer_thread.joinable()) {
            writer_thread.join();
        }

        std::swap(current_write_buf, next_write_buf);

        int local_write_count = 0;
        for (int i = 0; i < num_stars; ++i) {
            current_write_buf[local_write_count++] = {stars[i].x, stars[i].y, stars[i].z, stars[i].r, stars[i].g, stars[i].b, stars[i].type};
        }

        // Launch background thread to write this frame to SSD
        writer_thread = std::thread([out_file, current_write_buf, local_write_count]() {
            fwrite(current_write_buf, sizeof(PlaybackStar), local_write_count, out_file);
        });

        end_time = chrono::high_resolution_clock::now();
        chrono::duration<double> diff = end_time - start_time;
        total_time += diff.count();

        if (frame_count % 10 == 0) {
            cout << "Baked " << frame_count << "/" << total_frames << " frames (" << total_time << "s) | SimTime: " << simulation_time << endl;
        }
    }

    // Clean up last background write
    if (writer_thread.joinable()) {
        writer_thread.join();
    }

    fclose(out_file);
    cout << "Baking Complete in " << total_time << "s" << endl;

    if (use_gpu) {
        cuda_free_pinned_playback(host_buf_A);
        cuda_free_pinned_playback(host_buf_B);
    } else {
        delete[] host_buf_A;
        delete[] host_buf_B;
    }

#elif defined(MODE_PLAYBACK)
    long long file_size = 0;
    int num_playback_stars = 0;
    
    FILE* in_file = fopen("simulation.bin", "rb");
    if (in_file) {
        fseek(in_file, 0, SEEK_END);
        file_size = _ftelli64(in_file);
        fseek(in_file, 0, SEEK_SET);
        // We know we baked 2000 frames
        num_playback_stars = (file_size / 2000) / sizeof(PlaybackStar);
    } else {
        std::cerr << "simulation.bin not found! Please run bake.exe first." << std::endl;
        return 1;
    }
    
    // Override num_stars to the actual number of stars baked per frame
    num_stars = num_playback_stars;

    std::vector<PlaybackStar> read_buffer(num_stars);

    bool is_paused = false;
    bool space_was_pressed = false;

    struct KeyState {
        int key;
        double press_start_time = 0.0;
        double last_trigger_time = 0.0;
        bool was_pressed = false;
    };
    
    KeyState key_up{GLFW_KEY_UP};
    KeyState key_down{GLFW_KEY_DOWN};
    KeyState key_left{GLFW_KEY_LEFT};
    KeyState key_right{GLFW_KEY_RIGHT};

    double target_fps = 60.0;
    double last_update_time = glfwGetTime();

    auto check_key_trigger = [&](KeyState& state, double current_time, double repeat_delay, double initial_delay = 0.25) -> bool {
        bool is_pressed = (glfwGetKey(window, state.key) == GLFW_PRESS);
        bool triggered = false;
        
        if (is_pressed) {
            if (!state.was_pressed) {
                state.press_start_time = current_time;
                state.last_trigger_time = current_time;
                triggered = true;
            } else {
                double hold_duration = current_time - state.press_start_time;
                if (hold_duration >= initial_delay) {
                    double time_since_last_trigger = current_time - state.last_trigger_time;
                    if (time_since_last_trigger >= repeat_delay) {
                        state.last_trigger_time = current_time;
                        triggered = true;
                    }
                }
            }
        }
        state.was_pressed = is_pressed;
        return triggered;
    };

    while (!glfwWindowShouldClose(window)) {
        process_input(window);

        double current_time = glfwGetTime();

        // Control Keys
        bool space_is_pressed = (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS);
        if (space_is_pressed && !space_was_pressed) {
            is_paused = !is_paused;
            cout << (is_paused ? "PAUSED" : "PLAYING") << " (Speed: " << target_fps << " FPS)" << endl;
        }
        space_was_pressed = space_is_pressed;

        if (check_key_trigger(key_up, current_time, 0.05)) {
            target_fps = std::min(240.0, target_fps + 5.0);
            cout << "Speed increased to: " << target_fps << " FPS" << endl;
        }

        if (check_key_trigger(key_down, current_time, 0.05)) {
            target_fps = std::max(5.0, target_fps - 5.0);
            cout << "Speed decreased to: " << target_fps << " FPS" << endl;
        }

        bool step_forward = check_key_trigger(key_right, current_time, 0.03);
        bool step_backward = check_key_trigger(key_left, current_time, 0.03);

        // Determine if we should update to next frame
        double elapsed = current_time - last_update_time;
        bool should_update_frame = false;

        if (!is_paused) {
            if (elapsed >= (1.0 / target_fps)) {
                should_update_frame = true;
                last_update_time = current_time;
            }
        } else {
            if (step_forward) {
                should_update_frame = true;
            } else if (step_backward) {
                long frame_size = num_stars * sizeof(PlaybackStar);
                long current_pos = ftell(in_file);
                long new_pos = current_pos - 2 * frame_size;
                if (new_pos < 0) {
                    // Loop to end
                    fseek(in_file, 0, SEEK_END);
                    long file_size = ftell(in_file);
                    new_pos = file_size - frame_size;
                }
                fseek(in_file, new_pos, SEEK_SET);
                should_update_frame = true;
            }
        }

        if (should_update_frame) {
            size_t read_elements = fread(read_buffer.data(), sizeof(PlaybackStar), num_stars, in_file);
            if (read_elements != num_stars) {
                // Loop playback if EOF reached
                fseek(in_file, 0, SEEK_SET);
                fread(read_buffer.data(), sizeof(PlaybackStar), num_stars, in_file);
            }

            // Copy positions and colors to stars array for rendering
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < num_stars; ++i) {
                stars[i].x = read_buffer[i].x;
                stars[i].y = read_buffer[i].y;
                stars[i].z = read_buffer[i].z;
                stars[i].r = read_buffer[i].r;
                stars[i].g = read_buffer[i].g;
                stars[i].b = read_buffer[i].b;
                stars[i].type = read_buffer[i].padding;
            }

            update_vbo(stars, num_stars);
        }

        render_frame(num_stars);
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    fclose(in_file);
    cleanup_opengl();

#else
    while (!glfwWindowShouldClose(window)) {
        start_time = chrono::high_resolution_clock::now();
        
        process_input(window);

        float global_min = -10000.0f;
        float global_max = 10000.0f;

        if (use_gpu) {
            float actual_dt = cuda_physics_step(
                stars, num_stars, G, epsilon_sq, 0.01f,
                global_min, global_max, true // use_interop = true for LIVE
            );
            simulation_time += actual_dt;
        } else {
            float dt = 0.01f; // CPU uses fixed dt
            simulation_time += dt;
            // Calculate Bounds for CPU Octree only
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

            global_min = std::min({min_x, min_y, min_z});
            global_max = std::max({max_x, max_y, max_z});
            
#if defined(_MSC_VER)
            std::sort(stars, stars + num_stars, [global_min, global_max](const Star& a, const Star& b) {
                return get_morton_code(a.x, a.y, a.z, global_min, global_max) < 
                       get_morton_code(b.x, b.y, b.z, global_min, global_max);
            });
#else
            __gnu_parallel::sort(stars, stars + num_stars, [global_min, global_max](const Star& a, const Star& b) {
                return get_morton_code(a.x, a.y, a.z, global_min, global_max) < 
                       get_morton_code(b.x, b.y, b.z, global_min, global_max);
            });
#endif
            // Reset Arena and Build CPU Octree
            arena.reset();
            OctreeNode* root = arena.alloc<OctreeNode>();

            float max_span = std::max({max_x - min_x, max_y - min_y, max_z - min_z});
            root->center_x = (min_x + max_x) * 0.5f;
            root->center_y = (min_y + max_y) * 0.5f;
            root->center_z = (min_z + max_z) * 0.5f;
            root->half_width = max_span * 0.5f;
            root->is_leaf = true;
            root->start_star_index = 0;
            root->star_count = 0;

            uint32_t next_free_idx = 0;
            root->start_star_index = next_free_idx;
            next_free_idx += 32;

            for (uint32_t i = 0; i < num_stars; i++) {
                insert_star(root, i, stars, arena, leaf_star_indices, next_free_idx);
            }

            int num_nodes = arena.get_used_memory() / sizeof(OctreeNode);
            
            node_physics(root, root, stars, node_masses, node_com_x, node_com_y, node_com_z, leaf_star_indices);

            // Gravity Queries
            #pragma omp parallel for schedule(dynamic, 256)
            for (int i = 0; i < num_stars; ++i) {
                query_gravity(
                    root, stars[i], i, stars, 
                    node_masses, node_com_x, node_com_y, node_com_z, leaf_star_indices,
                    G, epsilon_sq, accels[i].ax, accels[i].ay, accels[i].az
                );
            }

            // Velocity / Position Integration
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < num_stars; ++i) {
                stars[i].vx += accels[i].ax * dt;
                stars[i].vy += accels[i].ay * dt;
                stars[i].vz += accels[i].az * dt;

                stars[i].x += stars[i].vx * dt;
                stars[i].y += stars[i].vy * dt;
                stars[i].z += stars[i].vz * dt;
            }
        }

        // Render
        if (!use_gpu) {
            update_vbo(stars, num_stars);
        }
        render_frame(num_stars);
        
        glfwSwapBuffers(window);
        glfwPollEvents();

        end_time = chrono::high_resolution_clock::now();
        chrono::duration<double> diff = end_time - start_time;
        if (frame_count % 60 == 0) {
            cout << "Frame " << frame_count << " processing time: " << diff.count() << "s | SimTime: " << simulation_time << " | FPS: " << 1.0/diff.count() << endl;
        }
        frame_count++;
    }

    cleanup_opengl();
    if (use_gpu) {
        cuda_unregister_vbo();
    }
#endif

    if (use_gpu) {
        cuda_cleanup();
    }

    delete[] leaf_star_indices;
    delete[] node_masses;
    delete[] node_com_x;
    delete[] node_com_y;
    delete[] node_com_z;

    #ifndef MODE_PLAYBACK
    if (use_gpu) {
        cuda_free_pinned_stars(stars);
    } else {
        delete[] stars;
    }
    #else
    delete[] stars;
    #endif

    return 0;
}