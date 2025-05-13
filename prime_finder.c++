#include <vector>
#include <cstdint>
#include <cmath>
#include <chrono>

#include <iostream>
#include <fstream>

std::vector<uint64_t> trial_division_naive(int size, uint64_t* time) {
    std::vector<uint64_t> primes;
    primes.reserve(size);
    primes.push_back(2);
    uint64_t candidate = 3;

    auto start_time = std::chrono::high_resolution_clock::now();

    while (primes.size() < size) {
        bool is_prime = true;

        for (size_t i = 0; i < primes.size(); i++) {
            uint64_t p = primes[i];
            if (p * p > candidate) break;
            if (candidate % p == 0) {
                is_prime = false;
                break;
            }
        }

        if (is_prime) {
            primes.emplace_back(candidate);
        }

        candidate += 1;
    }

    int64_t duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time).count();

    if(time){*time = duration;}
    return primes;
}

void test_trial_division_naive(int n){
    std::vector<uint64_t> f(n);
    std::vector<uint64_t> t(n);
    std::ofstream outfile("results/trial_division_naive.csv");

    outfile << "first n primes,";
    for(int i = 0; i < 10; i++){
        outfile << i << ", ";
    }
    outfile << "\n";

    for(int i = 0; i < n; i++){
        outfile << "10^" << i << ",";
        for(int j = 0; j < 10; j++){
            trial_division_naive(int(pow(10, i)), &t[i]);
            std::cout << "Time: " << t[i] << "us" << std::endl;
            outfile << t[i] << ",";
        }

        outfile << std::endl;
    }
    return;
}

struct wheel_return_t{
    std::vector<uint64_t> wheel_steps;
    std::vector<uint64_t> primes;
    uint64_t candidate;

    wheel_return_t(std::vector<uint64_t> w, std::vector<uint64_t> p, uint64_t l){
        wheel_steps = w;
        primes = p;
        candidate = l;
        return;
    }
};

wheel_return_t generate_wheel_steps(int wheel_size) {
    std::vector<uint64_t> primes = trial_division_naive(wheel_size + 1, nullptr);
    uint64_t last = primes.back();
    primes.pop_back();
    uint64_t limit = 1;
    for(auto x: primes){
        limit *= x;
    }
    std::vector<uint64_t> coprime_offsets;

    for (uint64_t i = 1; i < limit; ++i) {
        bool flag = true;
        for(auto x: primes){
            if(i % x == 0){
                flag = false;
            }
        }
        if(flag){
            coprime_offsets.push_back(i);
        }
    }

    std::vector<uint64_t> steps;
    for (size_t i = 1; i < coprime_offsets.size(); ++i) {
        steps.push_back(coprime_offsets[i] - coprime_offsets[i - 1]);
    }
    // wrap around to make the wheel repeat
    steps.push_back(wheel_size - coprime_offsets.back());

    return wheel_return_t(steps, primes, last);
}

std::vector<uint64_t> trial_division_naive_wheel(int size, int wheel_size, uint64_t* time) {
    wheel_return_t wheeldata = generate_wheel_steps(wheel_size);
    std::vector<uint64_t> primes = wheeldata.primes;
    std::vector<uint64_t> wheel_steps = wheeldata.wheel_steps;
    uint64_t candidate = wheeldata.candidate;
    
    primes.reserve(size);

    size_t wheel_index = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    while (primes.size() < size) {
        bool is_prime = true;

        for (size_t i = 0; i < primes.size(); i++) {
            uint64_t p = primes[i];
            if (p * p > candidate) break;
            if (candidate % p == 0) {
                is_prime = false;
                break;
            }
        }

        if (is_prime) {
            primes.emplace_back(candidate);
        }

        candidate += wheel_steps[wheel_index];
        wheel_index = (wheel_index + 1) % wheel_steps.size();
    }

    int64_t duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time).count();

    if(time){*time = duration;}
    return primes;
}

void test_trial_division_naive_wheel(int n, int wheel){
    std::vector<uint64_t> f(n);
    std::vector<uint64_t> t(n);
    for(int w = 7; w < wheel; w++){
        std::ofstream outfile("results/trial_division_naive_wheel" + std::to_string(w) + ".csv");

        outfile << "first n primes,";
        for(int i = 0; i < 10; i++){
            outfile << i << ", ";
        }
        outfile << "\n";

        for(int i = 0; i < n; i++){
            outfile << "10^" << i << ",";
            for(int j = 0; j < 10; j++){
                auto a = trial_division_naive_wheel(int(pow(10, i)), w, &t[i]);
                std::cout << "Time: " << t[i] << "us" << std::endl;
                outfile << t[i] << ",";
            }

            outfile << std::endl;
        }
    }
    return;
}

int main(){
    // test_trial_division_naive(8);
    // for(int i = 0; i <= 8; i++){
    //     test_trial_division_naive_wheel(8, i);
    // }
    
    return 0;
}