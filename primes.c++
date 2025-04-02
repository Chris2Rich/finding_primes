#include <iostream>
#include <csignal>
#include <vector>
#include <fstream>
#include <atomic>
#include <unistd.h>
#include <stdint.h>
#include <cuda_runtime.h>

std::atomic<bool> running(true);
uint64_t final_value = 0;

void handle_signal(int signum) {
    running = false;
}

//checks all primes, if any found, stop checking
__global__ void td_check_prime(uint64_t* primes, uint64_t candidate, uint64_t start_index, uint64_t chunk_size, int* found_flag) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int global_idx = start_index + idx;

    __shared__ int local_found_flag;

    if (threadIdx.x == 0) {
        local_found_flag = false;
    }
    __syncthreads();

    if (local_found_flag == 0 && global_idx < start_index + chunk_size) {
        if (candidate % primes[global_idx] == 0) {
            if (atomicExch(&local_found_flag, 1) == false) {
                atomicExch(found_flag, 1);
            }
        }
    }

    __syncthreads();

    if (local_found_flag != 0) {
        return;
    }
}

void trial_division(){
    std::vector<uint64_t> primes(100);
    primes[0] = 2;
    uint64_t candidate = 3;
    while(running){
        int found_flag = 0;
        td_check_prime(primes.data(), candidate, 0, 128, &found_flag);
        if (found_flag != 0){
            primes.push_back();
            final_value = candidate;
        }
        candidate++;
    }
}

int main(int argc, char** argv){
    signal(SIGTERM, handle_signal);
    signal(SIGINT, handle_signal);

    trial_division();

    std::ofstream file("result.txt");
    if (file.is_open()){
        file << final_value;
        file.close();
    } else {
        std::cerr << "Result not saved";
    }

    return 0;
}