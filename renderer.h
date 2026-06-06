#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "types.h"

GLFWwindow* init_opengl(int width, int height);
void process_input(GLFWwindow* window);
void setup_buffers(const Star* stars, size_t num_stars);
void update_vbo(const Star* stars, size_t num_stars);
void render_frame(size_t num_stars);
void cleanup_opengl();
