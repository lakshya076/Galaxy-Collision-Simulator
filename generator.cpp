#include "generator.h"

void spawn_disc(Star* stars, int& current_index, int num, std::mt19937& rng, uint8_t col_r, uint8_t col_g, uint8_t col_b) {
    float G = 1.0f;               
    float M = 100000.0f;          
    float a = 20.0f;              
    float b = 2.0f;

    float particle_mass = M / num;

    std::uniform_real_distribution<float> pos_dist(-100.0f, 100.0f);   
    std::uniform_real_distribution<float> z_dist(-10.0f, 10.0f);  
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    
    // Thermal velocity dispersion to keep disc stable
    std::normal_distribution<float> noise_dist(0.0f, 0.05f); 

    int spawned = 0;
    float max_density = (a + 3.0f * b) / (b * std::pow(a + b, 3.0f));

    while (spawned < num) {
        // Sample Cartesian uniformly to avoid density concentration at R=0
        float x = pos_dist(rng);
        float y = pos_dist(rng);
        float R = std::sqrt(x*x + y*y);

        if (R > 100.0f || R == 0.0f) continue;
        
        float z = z_dist(rng);

        float sqrt_z_b = std::sqrt(z*z + b*b);
        float a_sqrt = a + sqrt_z_b;

        float numerator = a * (R*R) + (a + 3.0f * sqrt_z_b) * (a_sqrt * a_sqrt);
        float denominator = (std::pow(R*R + (a_sqrt * a_sqrt), 2.5f)) * (std::pow(z*z + b*b, 1.5f));

        float density = numerator / denominator;

        if (prob_dist(rng) < (density / max_density)) {
            stars[current_index].x = x;
            stars[current_index].y = y;
            stars[current_index].z = z;

            stars[current_index].mass = particle_mass;
            stars[current_index].is_dm = false;
            stars[current_index].r = col_r;
            stars[current_index].g = col_g;
            stars[current_index].b = col_b;

            float vel_star = std::sqrt((G * M * R * R) / std::pow(R*R + a_sqrt * a_sqrt, 1.5f));
            
            // Orbital velocity + thermal dispersion
            stars[current_index].vx = vel_star * (-y / R) + noise_dist(rng) * vel_star;
            stars[current_index].vy = vel_star * (x / R) + noise_dist(rng) * vel_star;
            stars[current_index].vz = noise_dist(rng) * vel_star;

            current_index++;
            spawned++;
        }
    }
}

void spawn_bulge(Star* stars, int& current_index, int num, std::mt19937& rng, uint8_t col_r, uint8_t col_g, uint8_t col_b) {    
    float G = 1.0f;               
    float M = 20000.0f;           
    float a = 4.0f;               

    float particle_mass = M / num;

    std::uniform_real_distribution<float> u_dist(0.0f, 0.999f);
    std::uniform_real_distribution<float> theta_dist(0.0f, 2.0f * M_PI); 
    std::uniform_real_distribution<float> phi_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> v_dist(0.0f, 1.0f);

    for (int i = 0; i < num; i++) {
        float u = u_dist(rng);
        float sqrt_u = std::sqrt(u);
        float r = a * (sqrt_u / (1.0f - sqrt_u));

        float theta = theta_dist(rng);
        float cos_phi = phi_dist(rng);
        float sin_phi = std::sqrt(1.0f - cos_phi * cos_phi);

        stars[current_index].x = r * sin_phi * std::cos(theta);
        stars[current_index].y = r * sin_phi * std::sin(theta);
        stars[current_index].z = r * cos_phi;

        stars[current_index].mass = particle_mass; 
        stars[current_index].is_dm = false; 
        stars[current_index].r = col_r;
        stars[current_index].g = col_g;
        stars[current_index].b = col_b;

        float escape_velocity = std::sqrt((2.0f * G * M) / (r + a));
        float speed = escape_velocity * 0.4f * v_dist(rng); 

        float v_theta = theta_dist(rng);
        float v_cos_phi = phi_dist(rng);
        float v_sin_phi = std::sqrt(1.0f - v_cos_phi * v_cos_phi);

        stars[current_index].vx = speed * v_sin_phi * std::cos(v_theta);
        stars[current_index].vy = speed * v_sin_phi * std::sin(v_theta);
        stars[current_index].vz = speed * v_cos_phi;

        current_index++;
    }
}

void spawn_dark_matter(Star* stars, int& current_index, int num, std::mt19937& rng) {
    float G = 1.0f;
    float M_halo = 5000000.0f;
    float a = 100.0f;
    
    float particle_mass = M_halo / num;

    std::uniform_real_distribution<float> u_dist(0.0f, 0.999f);
    std::uniform_real_distribution<float> theta_dist(0.0f, 2.0f * M_PI);
    std::uniform_real_distribution<float> phi_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> v_dist(0.0f, 1.0f);

    for (int i = 0; i < num; i++) {
        float u = u_dist(rng);
        float sqrt_u = std::sqrt(u);
        float r = a * (sqrt_u / (1.0f - sqrt_u));

        float theta = theta_dist(rng);
        float cos_phi = phi_dist(rng);
        float sin_phi = std::sqrt(1.0f - cos_phi * cos_phi);

        stars[current_index].x = r * sin_phi * std::cos(theta);
        stars[current_index].y = r * sin_phi * std::sin(theta);
        stars[current_index].z = r * cos_phi;
        
        stars[current_index].mass = particle_mass; 
        stars[current_index].is_dm = true; 
        stars[current_index].r = 0;
        stars[current_index].g = 0;
        stars[current_index].b = 0;
        
        float escape_velocity = std::sqrt((2.0f * G * M_halo) / (r + a));
        float speed = escape_velocity * 0.4f * v_dist(rng); 

        float v_theta = theta_dist(rng);
        float v_cos_phi = phi_dist(rng);
        float v_sin_phi = std::sqrt(1.0f - v_cos_phi * v_cos_phi);

        stars[current_index].vx = speed * v_sin_phi * std::cos(v_theta);
        stars[current_index].vy = speed * v_sin_phi * std::sin(v_theta);
        stars[current_index].vz = speed * v_cos_phi;

        current_index++;
    }
}

void initial_setup(Star* stars, int num_dark_matter, int num_disc, int num_bulge, std::mt19937& rng) {
    int current_index = 0;
    
    // Galaxy A (Warm Palette)
    int start_A = current_index;
    spawn_dark_matter(stars, current_index, num_dark_matter, rng);
    spawn_disc(stars, current_index, num_disc, rng, 255, 100, 50);   // Orange/Red Disc
    spawn_bulge(stars, current_index, num_bulge, rng, 255, 255, 150); // Bright Yellow Core
    int end_A = current_index;
    
    // Clone Galaxy A to B
    int start_B = current_index;
    int particles_per_galaxy = end_A - start_A;
    for (int i = 0; i < particles_per_galaxy; i++) {
        stars[current_index] = stars[start_A + i];
        current_index++;
    }
    
    // Galaxy A: offset by x = -500, moving right
    for (int i = start_A; i < end_A; i++) {
        stars[i].x -= 500.0f;
        stars[i].vx += 50.0f;
    }
    
    // Galaxy B: offset by x = +500, moving left
    for (int i = start_B; i < current_index; i++) {
        stars[i].x += 500.0f;
        stars[i].vx -= 50.0f;
        
        // Recolor Galaxy B (Cool Palette)
        int local_idx = i - start_B;
        if (local_idx >= num_dark_matter) {
            if (local_idx < num_dark_matter + num_disc) {
                // Disc
                stars[i].r = 50; stars[i].g = 100; stars[i].b = 255; // Deep Blue
            } else {
                // Bulge
                stars[i].r = 200; stars[i].g = 255; stars[i].b = 255; // Cyan/White Core
            }
        }
    }
}