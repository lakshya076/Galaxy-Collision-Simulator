#pragma once

#include <random>
#include <cmath>
#include "types.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void spawn_disc(Star* stars, int& current_index, int num, std::mt19937& rng, uint8_t col_r, uint8_t col_g, uint8_t col_b);
void spawn_bulge(Star* stars, int& current_index, int num, std::mt19937& rng, uint8_t col_r, uint8_t col_g, uint8_t col_b);
void spawn_dark_matter(Star* stars, int& current_index, int num, std::mt19937& rng);
void initial_setup(Star* stars, int num_dark_matter, int num_disc, int num_bulge, std::mt19937& rng);
