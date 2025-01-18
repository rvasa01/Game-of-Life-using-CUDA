# CUDA-based John Conway's Game of Life

This project implements **John Conway's Game of Life** using **CUDA** for parallel computation. It explores the use of CUDA memory types (normal, pinned, and managed) to accelerate the simulation of the cellular automaton.

## Features
- **Game Objective**:
  - Simulates Conway's Game of Life on a 2D grid where cells are either alive or dead.
  - Cells evolve across generations based on the following rules:
    - A live cell remains alive if it has 2 or 3 live neighbors.
    - A dead cell becomes alive if it has exactly 3 live neighbors.
- **CUDA Integration**:
  - Implements the Game of Life using CUDA with three memory allocation types:
    - **Normal**: Standard GPU memory.
    - **Pinned**: Pinned memory for faster host-to-device transfers.
    - **Managed**: Unified memory shared between host and device.
- **Command-Line Arguments**:
  - `-n`: Threads per block (must be a multiple of 32; defaults to 32).
  - `-c`: Cell size (square cells; defaults to 5).
  - `-x`: Window width (defaults to 800).
  - `-y`: Window height (defaults to 600).
  - `-t`: Memory allocation type (`NORMAL`, `PINNED`, or `MANAGED`; defaults to `NORMAL`).
  - Example: `./Lab4 -n 64 -c 5 -x 800 -y 600 -t PINNED`.
- **Graphics**:
  - Displays the current state of the grid:
    - White cells for alive.
    - Black cells for dead (not rendered).
- **Console Output**:
  - Reports processing time (in microseconds) for the last 100 generations for each memory type.

## How to Run
1. Clone the repository and ensure you have CUDA installed on your system.
2. Compile the program using the provided `CMakeLists.txt`.
3. Run the executable with the desired command-line arguments.
4. View the simulation in the graphical window.
5. Press the `Esc` key to exit the program.

