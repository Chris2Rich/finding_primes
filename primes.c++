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
__global__ void td_check_prime(uint64_t* primes, uint64_t candidate, int N, bool* found_flag) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        if (*found_flag){
            return;
        }

        if (candidate % primes[idx] == 0){
            atomicExch(found_flag, true);
            return;
        }
    }
}

void trial_division(){
    std::vector<uint64_t> primes(100);
    primes[0] = 2;
    while(running){
        if (){

        } else {
            final_value = number
        }
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