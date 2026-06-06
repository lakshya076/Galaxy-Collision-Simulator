# 3D Galaxy Collision Simulator

A high-performance, custom-built 3D Galaxy Collision Simulator written in pure C++ (using the C++17 Standard Library only). This project simulates the gravitational interaction and collision of two stable galaxies (comprising stellar discs, bulges, and massive dark matter halos) using a highly optimized Barnes-Hut octree.

---

## Project Overview & Roadmap

This project is built incrementally following the roadmap:

- **Foundation & Spatial DB Integration** - Standardized particle representations, memory pooling, and importing stellar catalogs from spatial data sources. This was completed in another project first stored as **[Spatial_DB Repository](https://github.com/lakshya076/Spatial_DB)**. Currently instead of loading external data, mathematical models are used to generate two galaxies, the full generated data is stored in the Octree format to optimize Barnes-Hut later.
- **Initial Conditions Generator** - Mathematical generation of stable galaxies using
  - Miyamoto-Nagai stellar discs
  - Hernquist bulges
  - Dark Matter Halos
- **Barnes-Hut Gravity Approximation and Leapfrog Integration** - Bottom-up mass aggregation and fast gravitational force calculation using the $\theta = \frac{s}{d}$ approximation. Then the temporal integrator calculates the updated velocities and positions of each star over time steps ($\Delta t$).
- **Live OpenGL Graphics Pipeline** - Real-time visualization using GLFW/GLEW with a free-flying WASD camera. The `PlaybackStar` memory layout streams directly into OpenGL Vertex Buffer Objects (VBOs) with zero overhead.
- **Offline Physics Baking** - Decoupling the engine into a background Physics Baker (dumps binary data to SSD) and a GPU Playback Viewer for an adjustable FPS viewing of extremely heavy simulations.
- **GPU/CUDA Optimization** - CPU based simulation cost around 2 seconds per frame generation. CUDA optimizations were made to run the heavy compute directly onto the GPU to reduce this time to ~0.4 seconds per frame generation.

---

## 🛠️ Architecture & Constraints

To achieve extreme real-time simulation performance, the engine adheres to strict hardware-friendly memory layouts and caching constraints:

### 1. The Memory Pool (`ArenaAllocator`)
Instead of standard dynamic allocations (`new`/`delete`) which cause memory fragmentation and latency, the simulator relies on a custom `ArenaAllocator`. It provisions contiguous blocks of RAM instantly. The arena is completely wiped (pointer reset to 0) and rebuilt every single frame.

### 2. High-Density Cache-Line Octree (`OctreeNode`)
`OctreeNode` structs are strictly packed and aligned to exactly 32 bytes (`alignas(32)`). This guarantees that reading a node loads exactly half of an L1 cache-line, preventing cache misses and doubling cache density. 
* The bounds are mathematically simplified using a `center` coordinate and `half_width` instead of standard min/max bounding boxes.
* To maintain this 32-byte limit, **no physics data (like mass or center of mass) is stored inside the node struct itself**. 

### 3. The Indirection Array
Instead of limiting leaf nodes to a small array of stars, the engine uses a global Indirection Array. Leaf nodes store a 32-bit `start_star_index`, allowing them to hold up to 32 stars without increasing the node struct size. This keeps the octree incredibly shallow, eliminating pointer-chasing latency.

### 4. Star Struct (Array of Structures)
All stellar particles (visible disc stars, bulge stars, and dark matter macro-particles) are stored in a unified flat array:
```cpp
struct Star {
    float x, y, z;    // 3D position
    float vx, vy, vz; // 3D velocity
    float mass;       // Particle mass
    bool is_dm;       // Dark matter flag (is_dm = true skips rendering)
};
```

### 5. Node Physics (Structure of Arrays)
Because the `OctreeNode` cannot exceed 32 bytes, all node-level physics data (such as total mass and 3D Center of Mass) is stored in parallel arrays (`float* node_masses`, etc.) and accessed using the node's index.

### 6. Memory Locality & Prefetching
* **Morton Z-Order Curve Sorting:** The `stars` array is parallel-sorted using GCC's `__gnu_parallel::sort` according to a 64-bit interleaved Morton Code before tree construction. This ensures OpenMP threads process physically adjacent subsets of the tree, maximizing spatial locality.
* **Manual L1 Memory Prefetching:** The hottest direct-gravity math loop utilizes compiler intrinsics (`__builtin_prefetch`) to actively pull the *next* star's data into the L1 cache while the FPU computes the inverse square root math for the *current* star.

### 7. GPU CUDA Architecture
* **Structure of Arrays (SoA):** Before launching the physics kernels, the `Star` array (AoS) is transposed into separate parallel arrays (`d_x`, `d_y`, `d_z`, etc.) in VRAM. This guarantees perfectly coalesced memory access across 32-thread warps.
* **Non-Recursive Octree Traversal:** The GPU traverses the Barnes-Hut octree using a highly optimized iterative stack stored locally per-thread. This avoids recursive function calls which would otherwise cause massive local memory allocation and slow down the GPU.
* **Active Node Masking:** To prevent the GPU from wasting cycles evaluating empty regions of space, the CPU tree builder populates an 8-bit `active_mask` in each `OctreeNode`. The GPU checks this bitmask and only pushes populated children onto its traversal stack.
* **Zero-Copy VBO Interop:** In Live mode, CUDA registers and maps the OpenGL Vertex Buffer Objects (VBOs) directly. The calculated physics positions are written straight into the graphics memory buffer. This eliminates the need to transfer the positions back to the CPU over the PCIe bus to render them, massively improving framerates.

---

## 🧪 Mathematical Formulations (Initial Conditions)

The initial galaxies are generated using the following astronomical density profiles:
* **Miyamoto-Nagai Disc:** Used to spawn visible stars in a flattened stellar disk with circular orbital velocities.
* **Hernquist Bulge:** Generates a tightly packed galactic core with random velocity dispersions.
* **Hernquist Dark Matter Halo:** Spawns massive, diffuse dark matter macro-particles (`is_dm = true`) to act as the gravitational glue stabilizing the galaxies.
* **Collision Setup:** The generator constructs two identical stable galaxies, applies a spatial offset (e.g., $\pm 500$ units on the X-axis), adds opposing translational velocity vectors to set them on a collision course, and merges them into the flat `stars` array.

---

## ⏱️ Performance Benchmarks

Simulating the gravitational physics for **300,000 particles** using the $O(N \log N)$ Barnes-Hut algorithm is extremely heavy. The system was engineered across several major revisions to maximize framerates. 

Here is the evolution of the per-frame generation time (lower is better):

| Implementation Phase | Frame Gen Time | Effective FPS | Notes |
| :--- | :--- | :--- | :--- |
| **1. Baseline CPU (Single Threaded)** | ~39.96 s | ~0.025 FPS | Initial Conditions.|
| **1. Baseline CPU (OpenMP)** | ~2.00 s | ~0.5 FPS | Achieved parallelization using OpenMP.|
| **2. Initial CUDA GPU Port** | ~0.45 s | ~2.2 FPS | Naive GPU physics computation using standard data layouts. |
| **3. Heavily Optimized GPU** | ~0.30 s | ~3.3 FPS | **Current State.** 30% speedup achieved by pre-computing Morton Sorting on CPU (`0.02s`), switching to SoA VRAM layouts, storing the traversal stack in 64KB `__shared__` memory, caching reads with `__ldg`, and skipping empty space with the `active_mask`.<br><br>*(Note: The actual GPU math now only takes **0.05 seconds**. The total time is currently bottlenecked by the sequential CPU octree builder).* |
| **4. Playback Mode (Zero-Physics)** | ~0.007 s | **Adjustable FPS** | Bypasses all physics and streams the baked `simulation.bin` directly into the GPU's VRAM. Limited only by disk read speed and monitor refresh rate. |

---

## 🚀 Getting Started

### Prerequisites

#### For CPU-Only (GCC) Build:
* Windows 11
* A C++17 compliant compiler (e.g., `g++` on Linux or [MSYS2](https://www.msys2.org/)/MinGW64 on Windows).
* OpenMP support.
* Modern OpenGL development libraries (GLFW, GLEW, GLM).

#### For GPU-Accelerated (CUDA) Build:
* **NVIDIA GPU** with CUDA support.
* **NVIDIA CUDA Toolkit** (containing the `nvcc` compiler). [Download Here](https://developer.nvidia.com/cuda-downloads)
* **Microsoft Visual Studio (MSVC)**: On Windows, `nvcc` strictly requires the MSVC host compiler (`cl.exe`) to preprocess host-side code. [Download Here](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170)
* Modern OpenGL development libraries (GLFW, GLEW, GLM).

On MSYS2 (Windows), install the OpenGL dependencies:
```bash
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-glfw mingw-w64-x86_64-glew mingw-w64-x86_64-glm
```

---

### Compiling and Running

This simulator is split into three decoupled execution modes, allowing you to bake incredibly heavy physics offline and play them back later in real-time. You can compile either the GPU-accelerated version (utilizing CUDA) or the CPU-only version (utilizing OpenMP).

> [!IMPORTANT]
> **Customizing Paths for Your Machine:**
> You **must** adjust the folder paths in the compilation commands below to match your local installation:
> 1. `-ccbin "D:\VisualStudio\VC\Tools\MSVC\..."`: Point this to the absolute directory containing your local MSVC host compiler `cl.exe`.
> 2. `-I"C:\msys64\mingw64\include"` and library paths (`"C:\msys64\mingw64\lib\libglfw3.dll.a"` / `"C:\msys64\mingw64\lib\libglew32.dll.a"`): Point these to your local MinGW or header/library folder where GLFW, GLEW, and GLM are installed.
> 3. `-arch=sm_86`: Adjust to target your GPU's actual Compute Capability architecture (e.g., `sm_86` for Ampere/RTX 30xx, `sm_89` for Ada Lovelace/RTX 40xx, `sm_75` for Turing/RTX 20xx).

#### 1. Live Mode (Normal Simulation & Render)
Runs the physics engine and immediately renders each frame. Best for real-time visualization of moderate to large particle counts.
* **GPU CUDA Build (Recommended):**
  ```bash
  nvcc -O3 -arch=sm_86 -ccbin "D:\VisualStudio\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64" -Xcompiler "/openmp /fp:fast /arch:AVX2" .\main.cpp .\generator.cpp .\engine.cpp .\gravity_aggregator.cpp .\renderer.cpp .\cuda_engine.cu -o main.exe -I"C:\msys64\mingw64\include" "C:\msys64\mingw64\lib\libglfw3.dll.a" "C:\msys64\mingw64\lib\libglew32.dll.a" -lopengl32 -lgdi32 -luser32 -lshell32
  $env:PATH += ";C:\msys64\mingw64\bin"; .\main.exe
  ```
  *(Pass the `--cpu` argument at runtime to bypass the GPU and run on OpenMP CPU fallback.)*
* **CPU-Only GCC Build:**
  ```bash
  g++ -std=c++17 -O3 -fopenmp -ffast-math -march=native .\main.cpp .\generator.cpp .\engine.cpp .\gravity_aggregator.cpp .\renderer.cpp -o main_cpu.exe -I"C:\msys64\mingw64\include" -L"C:\msys64\mingw64\lib" -lglfw3 -lglew32 -lopengl32 -lgdi32
  $env:PATH += ";C:\msys64\mingw64\bin"; .\main_cpu.exe
  ```

#### 2. Bake Mode (Offline Physics Baker)
Calculates physics as fast as possible in the terminal (window rendering is disabled) and dumps the visible stars to `simulation.bin` using the compact 16-byte `PlaybackStar` format. For a 1000 frame bake, this file reaches around **3.2 GB** in size (down from the uncompressed **9.2 GB** thanks to dark matter filtering and attribute stripping).
* **GPU CUDA Build (Recommended):**
  ```bash
  nvcc -O3 -arch=sm_86 -DMODE_BAKE -ccbin "D:\VisualStudio\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64" -Xcompiler "/openmp /fp:fast /arch:AVX2" .\main.cpp .\generator.cpp .\engine.cpp .\gravity_aggregator.cpp .\renderer.cpp .\cuda_engine.cu -o bake.exe -I"C:\msys64\mingw64\include" "C:\msys64\mingw64\lib\libglfw3.dll.a" "C:\msys64\mingw64\lib\libglew32.dll.a" -lopengl32 -lgdi32 -luser32 -lshell32
  $env:Path += ";C:\msys64\mingw64\bin"; .\bake.exe <num_frames>
  ```
  *(Pass `--cpu` as the second argument, e.g. `.\bake.exe 1000 --cpu`, to bake using the CPU engine.)*
* **CPU-Only GCC Build:**
  ```bash
  g++ -std=c++17 -O3 -fopenmp -ffast-math -march=native -DMODE_BAKE .\main.cpp .\generator.cpp .\engine.cpp .\gravity_aggregator.cpp .\renderer.cpp -o bake_cpu.exe -I"C:\msys64\mingw64\include" -L"C:\msys64\mingw64\lib" -lglfw3 -lglew32 -lopengl32 -lgdi32
  $env:Path += ";C:\msys64\mingw64\bin"; .\bake_cpu.exe <num_frames>
  ```

#### 3. Playback Mode (Zero-Physics Viewer)
Disables the physics engine entirely. Initializes OpenGL and rapidly streams the pre-baked frames from `simulation.bin` directly into the VRAM at adjustable FPS.
* **GPU CUDA Build (Recommended):**
  ```bash
  nvcc -O3 -arch=sm_86 -DMODE_PLAYBACK -ccbin "D:\VisualStudio\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64" -Xcompiler "/openmp /fp:fast /arch:AVX2" .\main.cpp .\generator.cpp .\engine.cpp .\gravity_aggregator.cpp .\renderer.cpp .\cuda_engine.cu -o play.exe -I"C:\msys64\mingw64\include" "C:\msys64\mingw64\lib\libglfw3.dll.a" "C:\msys64\mingw64\lib\libglew32.dll.a" -lopengl32 -lgdi32 -luser32 -lshell32
  $env:PATH += ";C:\msys64\mingw64\bin"; .\play.exe
  ```
* **CPU-Only GCC Build:**
  ```bash
  g++ -std=c++17 -O3 -fopenmp -ffast-math -march=native -DMODE_PLAYBACK .\main.cpp .\generator.cpp .\engine.cpp .\gravity_aggregator.cpp .\renderer.cpp -o play_cpu.exe -I"C:\msys64\mingw64\include" -L"C:\msys64\mingw64\lib" -lglfw3 -lglew32 -lopengl32 -lgdi32
  $env:PATH += ";C:\msys64\mingw64\bin"; .\play_cpu.exe
  ```

### Controls (Live & Playback Mode)
- **Mouse Look:** Pitch and Yaw rotation.
- **W / S:** Move Forward / Backward.
- **A / D:** Strafe Left / Right.
- **Mouse Scroll:** Fast Forward / Backward Zoom.
- **SPACE:** Pause / Resume playback (Playback mode only).
- **UP / DOWN:** Increase / Decrease speed (target FPS) (Playback mode only).
- **LEFT / RIGHT:** Step 1 frame Backward / Forward (when paused in Playback mode).
- **ESC:** Exit Simulation.
