/*
Author: Rahil Vasa
Class: ECE4122
Last Date Modified: 11/07/2024
Description:
Parallel processing code for Game of Life with CUDA. 
*/

#include <SFML/Graphics.hpp>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <unistd.h>
#include <chrono>
#include <cuda_runtime.h>
#include <string>


int WINDOW_WIDTH = 800, WINDOW_HEIGHT = 600, PIXEL_SIZE = 5, NUM_THREADS = 32;
std::string PROCESSING_TYPE = "NORMAL";
int GRID_WIDTH, GRID_HEIGHT;

// CUDA kernel to update the grid based on Game of Life rules
__global__ void gameOfLifeKernel(uint8_t* grid_current, uint8_t* grid_next, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int y = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (x <= width && y <= height) {
        int idx = y * (width + 2) + x;

        // Count the number of alive neighbors
        int neighbors = grid_current[idx - (width + 2) - 1] + grid_current[idx - (width + 2)] + grid_current[idx - (width + 2) + 1]
                      + grid_current[idx - 1] + grid_current[idx + 1]
                      + grid_current[idx + (width + 2) - 1] + grid_current[idx + (width + 2)] + grid_current[idx + (width + 2) + 1];

        // Apply the Game of Life rules to determine the cell's next state
        grid_next[idx] = (grid_current[idx]) ? ((neighbors == 2) || (neighbors == 3)) : (neighbors == 3);
    }
}

// Function to initialize the grid with random alive or dead cells
void seedRandomGrid(uint8_t* grid, int width, int height) {
    std::srand(static_cast<unsigned>(std::time(nullptr)));  
    for (int y = 1; y <= height; ++y) {  
        for (int x = 1; x <= width; ++x) {  
            grid[y * (width + 2) + x] = std::rand() % 2;  
        }
    }
}

// Function to update the grid using normal memory allocation
void updateGridNormal(uint8_t* h_grid_current, uint8_t* h_grid_next,
                      uint8_t* d_grid_current, uint8_t* d_grid_next,
                      size_t size, dim3 threadsPerBlock, dim3 blocksPerGrid) {
    // Copy current grid from host to device memory
    cudaMemcpy(d_grid_current, h_grid_current, size, cudaMemcpyHostToDevice);

    // Launch the CUDA kernel to compute the next grid state
    gameOfLifeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_grid_current, d_grid_next, GRID_WIDTH, GRID_HEIGHT);
    cudaDeviceSynchronize();

    // Copy the computed next grid state back to host memory
    cudaMemcpy(h_grid_next, d_grid_next, size, cudaMemcpyDeviceToHost);
}

// Function to update the grid using pinned memory allocation
void updateGridPinned(uint8_t* h_grid_current, uint8_t* h_grid_next,
                      uint8_t* d_grid_current, uint8_t* d_grid_next,
                      size_t size, dim3 threadsPerBlock, dim3 blocksPerGrid) {
    // Asynchronously copy current grid from host to device memory
    cudaMemcpyAsync(d_grid_current, h_grid_current, size, cudaMemcpyHostToDevice);

    // Launch the CUDA kernel to compute the next grid state
    gameOfLifeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_grid_current, d_grid_next, GRID_WIDTH, GRID_HEIGHT);
    cudaDeviceSynchronize();

    // Asynchronously copy the computed next grid state back to host memory
    cudaMemcpyAsync(h_grid_next, d_grid_next, size, cudaMemcpyDeviceToHost);
}

// Function to update the grid using managed memory allocation
void updateGridManaged(uint8_t* grid_current, uint8_t* grid_next,
                       dim3 threadsPerBlock, dim3 blocksPerGrid) {
    // Launch the CUDA kernel to compute the next grid state
    gameOfLifeKernel<<<blocksPerGrid, threadsPerBlock>>>(grid_current, grid_next, GRID_WIDTH, GRID_HEIGHT);
    cudaDeviceSynchronize();
}

int main(int argc, char* argv[]) {
    int opt;
    while ((opt = getopt(argc, argv, "n:c:x:y:t:")) != -1) {
        switch (opt) {
            case 'n':
                NUM_THREADS = std::max(32, (std::atoi(optarg) / 32) * 32);
                break;
            case 'c':
                PIXEL_SIZE = std::max(1, std::atoi(optarg));
                break;
            case 'x':
                WINDOW_WIDTH = std::atoi(optarg);
                break;
            case 'y':
                WINDOW_HEIGHT = std::atoi(optarg);
                break;
            case 't':
                PROCESSING_TYPE = optarg;
                break;
        }
    }

    GRID_WIDTH = WINDOW_WIDTH / PIXEL_SIZE;
    GRID_HEIGHT = WINDOW_HEIGHT / PIXEL_SIZE;

    int threadsPerBlockX = 32;
    int threadsPerBlockY = NUM_THREADS / 32;
    if (threadsPerBlockY < 1) threadsPerBlockY = 1;
    dim3 threadsPerBlock(threadsPerBlockX, threadsPerBlockY);

    dim3 blocksPerGrid((GRID_WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (GRID_HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sf::RenderWindow window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "Game of Life");
    window.setFramerateLimit(60); 


    size_t size = (GRID_WIDTH + 2) * (GRID_HEIGHT + 2) * sizeof(uint8_t);
    uint8_t *grid_current, *grid_next;  // Host grids
    uint8_t *d_grid_current = nullptr, *d_grid_next = nullptr;  // Device grids


    if (PROCESSING_TYPE == "NORMAL") {
        // Allocate host memory
        grid_current = new uint8_t[(GRID_WIDTH + 2)*(GRID_HEIGHT + 2)];
        grid_next = new uint8_t[(GRID_WIDTH + 2)*(GRID_HEIGHT + 2)];
        seedRandomGrid(grid_current, GRID_WIDTH, GRID_HEIGHT);

        // Allocate device memory once
        cudaMalloc(&d_grid_current, size);
        cudaMalloc(&d_grid_next, size);
    } else if (PROCESSING_TYPE == "PINNED") {
        // Allocate pinned host memory
        cudaMallocHost(&grid_current, size);
        cudaMallocHost(&grid_next, size);
        seedRandomGrid(grid_current, GRID_WIDTH, GRID_HEIGHT);

        cudaMalloc(&d_grid_current, size);
        cudaMalloc(&d_grid_next, size);
    } else if (PROCESSING_TYPE == "MANAGED") {
        // Allocate unified memory accessible by both host and device
        cudaMallocManaged(&grid_current, size);
        cudaMallocManaged(&grid_next, size);
        seedRandomGrid(grid_current, GRID_WIDTH, GRID_HEIGHT);
    }

    int generation_count = 0; 
    long long delta_t = 0;     // Time accumulator

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed ||
               (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Escape))
                window.close();
        }

        // Start timing the grid update
        auto start = std::chrono::high_resolution_clock::now();

        if (PROCESSING_TYPE == "NORMAL"){
            updateGridNormal(grid_current, grid_next, d_grid_current, d_grid_next, size, threadsPerBlock, blocksPerGrid);
        }else if (PROCESSING_TYPE == "PINNED"){
            updateGridPinned(grid_current, grid_next, d_grid_current, d_grid_next, size, threadsPerBlock, blocksPerGrid);
        }else if (PROCESSING_TYPE == "MANAGED"){
            updateGridManaged(grid_current, grid_next, threadsPerBlock, blocksPerGrid);
        }
        // End timing the grid update
        auto end = std::chrono::high_resolution_clock::now();
        delta_t += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        generation_count++;

        if (generation_count == 100) {
            std::cout << "100 generations took " << delta_t << " microseconds with ";
            std::cout << NUM_THREADS << " threads per block using " << PROCESSING_TYPE << " memory allocation.\n";
            generation_count = 0;
            delta_t = 0;
        }

        std::swap(grid_current, grid_next);
        if (d_grid_current && d_grid_next) {
            std::swap(d_grid_current, d_grid_next);
        }

        window.clear(sf::Color::Black);

        // Create a vertex array to hold cell shapes
        sf::VertexArray cells(sf::Triangles);

        // Loop through the grid and add alive cells to the vertex array
        for (int y = 1; y <= GRID_HEIGHT; ++y) {
            for (int x = 1; x <= GRID_WIDTH; ++x) {
                if (grid_current[y * (GRID_WIDTH + 2) + x]) {
                    float px = (x - 1) * PIXEL_SIZE;

                    float py = (y - 1) * PIXEL_SIZE;
                    cells.append(sf::Vertex(sf::Vector2f(px, py), sf::Color::White));
                    cells.append(sf::Vertex(sf::Vector2f(px + PIXEL_SIZE, py), sf::Color::White));
                    cells.append(sf::Vertex(sf::Vector2f(px + PIXEL_SIZE, py + PIXEL_SIZE), sf::Color::White));

                    cells.append(sf::Vertex(sf::Vector2f(px, py), sf::Color::White));
                    cells.append(sf::Vertex(sf::Vector2f(px + PIXEL_SIZE, py + PIXEL_SIZE), sf::Color::White));
                    cells.append(sf::Vertex(sf::Vector2f(px, py + PIXEL_SIZE), sf::Color::White));
                }
            }
        }

        window.draw(cells);
        window.display();
    }

    // Free allocated memory 
    if (PROCESSING_TYPE == "NORMAL") {
        delete[] grid_current; delete[] grid_next;
        cudaFree(d_grid_current); 
        cudaFree(d_grid_next);
    } else if (PROCESSING_TYPE == "PINNED") {
        cudaFreeHost(grid_current); 
        cudaFreeHost(grid_next);
        cudaFree(d_grid_current); 
        cudaFree(d_grid_next);
    } else if (PROCESSING_TYPE == "MANAGED") {
        cudaFree(grid_current); 
        cudaFree(grid_next);
    }

    return 0;
}
