#include <bits/stdc++.h>

std::vector<uint64_t> trial_division_naive(uint64_t size, uint64_t* time) {
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

        candidate += 2;
    }

    int64_t duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time).count();

    if(time){*time = duration;}
    return primes;
}

void test_trial_division_naive(uint64_t n){
    std::vector<uint64_t> f(n);
    std::vector<uint64_t> t(n);
    std::ofstream outfile("times/trial_division_naive.csv");
    std::ofstream res("results/restrial_division_naive.res");

    outfile << "first n primes,";
    for(uint64_t i = 0; i < 5; i++){
        outfile << i << ", ";
    }
    outfile << "\n";

    for(uint64_t i = 1; i < n; i++){
        outfile << "10^" << i << ",";
        for(uint64_t j = 0; j < 5; j++){
            trial_division_naive(uint64_t(pow(10, i)), &t[i]);
            // for(auto x: trial_division_naive(uint64_t(pow(10, i)), &t[i])){res << x << "\n";}
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

wheel_return_t generate_wheel_steps(uint64_t wheel_size) {
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
    steps.push_back(limit + coprime_offsets.front() - coprime_offsets.back());

    return wheel_return_t(steps, primes, last);
}

std::vector<uint64_t> trial_division_wheel(uint64_t size, uint64_t wheel_size, uint64_t* time) {
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

void test_trial_division_wheel(uint64_t n, uint64_t wheel){
    std::vector<uint64_t> f(n);
    std::vector<uint64_t> t(n);
    for(uint64_t w = 2; w < wheel; w++){
        std::ofstream outfile("times/trial_division_wheel" + std::to_string(w) + ".csv");
        std::ofstream res("results/restrial_division_wheel" + std::to_string(w) + ".res");

        outfile << "first n primes,";
        for(uint64_t i = 0; i < 5; i++){
            outfile << i << ", ";
        }
        outfile << "\n";

        for(uint64_t i = 1; i < n; i++){
            outfile << "10^" << i << ",";
            for(uint64_t j = 0; j < 5; j++){
                trial_division_wheel(uint64_t(pow(10, i)), w, &t[i]);
                // for(auto x: trial_division_wheel(uint64_t(pow(10, i)), w, &t[i])){res << x << "\n";}
                std::cout << "Time: " << t[i] << "us" << std::endl;
                outfile << t[i] << ",";
            }

            outfile << std::endl;
        }
    }
    return;
}

uint64_t expmod(uint64_t x, uint64_t n, uint64_t m){
    if(n == 0){return 1;}
    uint64_t y = 1;
    while(n > 1){
        if(n & 1){
            y = __uint128_t(x * y) % m;
            n -= 1;
        }
        x = __uint128_t(x * x) % m;
        n >>= 1;
    }
    return __uint128_t(x * y) % m;
}

bool miller_rabin_logic(uint64_t candidate, uint64_t s, uint64_t d, uint64_t k, std::mt19937_64 gen, std::uniform_int_distribution<uint64_t> dis){
    for(uint64_t i = 0; i < k; i++){
        uint64_t a = (dis(gen) % (candidate - 3)) + 1;
        uint64_t x = expmod(a, d, candidate);
        uint64_t y = 0;
        for(int64_t j = 0; j < s; j++){
            y = __uint128_t(x*x) % candidate;
            if(y == 1 && x != 1 && x != (candidate - 1)){
                return false;
            }
            x = y;
        }
        if(y != 1){
            return false;
        }
    }
    return true;
}

//implemented such that if the last value returned is correct, every preceeding number is correct
// as there are no false negatives and a false positive would change the last number due to an early return
std::vector<uint64_t> miller_rabin_naive(uint64_t size, uint64_t k, uint64_t* time){

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dis;
    std::vector<uint64_t> primes = {2,3};
    primes.reserve(size);

    auto start_time = std::chrono::high_resolution_clock::now();
    uint64_t candidate = 5;

    while(primes.size() < size){
        uint64_t s = __builtin_ctzll(candidate - 1);
        uint64_t d = (candidate - 1) >> s;

        if(miller_rabin_logic(candidate, s, d, k, gen , dis)){
            primes.emplace_back(candidate);
        }

        candidate += 2;
    }

    int64_t duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time).count();

    if(time){*time = duration;}
    return primes;
}

void test_miller_rabin_naive(uint64_t n){
    std::vector<uint64_t> f(n);
    std::vector<uint64_t> t(n);
    std::ofstream outfile("times/miller_rabin_naive.csv");
    std::ofstream res("results/resmiller_rabin_naive.res");

    outfile << "first n primes,";
    for(uint64_t i = 0; i < 5; i++){
        outfile << i << ", ";
    }
    outfile << "\n";

    for(uint64_t i = 1; i < n; i++){
        outfile << "10^" << i << ",";
        for(uint64_t j = 0; j < 5; j++){
            //P(false positive) = (1/4)^k. chance of error in ENTIRE test should be 0.1%
            // total generated is 1111111110
            //k = ln(0.001 * (1/1111111110))/ln(1/4) = 20.007 rounds
            miller_rabin_naive(uint64_t(pow(10, i)), 20, &t[i]);
            // for(auto x: miller_rabin_naive(uint64_t(pow(10, i)), 20, &t[i])){res << x << "\n";}
            std::cout << "Time: " << t[i] << "us" << std::endl;
            outfile << t[i] << ",";
        }

        outfile << std::endl;
    }
    return;
}

std::vector<uint64_t> miller_rabin_wheel(uint64_t size, uint64_t k, uint64_t wheel_size, uint64_t* time){
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dis;

    wheel_return_t wheeldata = generate_wheel_steps(wheel_size);
    std::vector<uint64_t> primes = wheeldata.primes;
    std::vector<uint64_t> wheel_steps = wheeldata.wheel_steps;
    uint64_t candidate = wheeldata.candidate;
    
    primes.reserve(size);
    size_t wheel_index = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    while(primes.size() < size){
        uint64_t s = __builtin_ctzll(candidate - 1);
        uint64_t d = (candidate - 1) >> s;

        if(miller_rabin_logic(candidate, s, d, k, gen , dis)){
            primes.emplace_back(candidate);
        }
                
        candidate += wheel_steps[wheel_index];
        wheel_index = (wheel_index + 1) % wheel_steps.size();
    }

    int64_t duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time).count();

    if(time){*time = duration;}
    return primes;
}

void test_miller_rabin_wheel(uint64_t n, uint64_t wheel_size){
    for(int w = 2; w < wheel_size; w++){
        std::vector<uint64_t> f(n);
        std::vector<uint64_t> t(n);
        std::ofstream outfile("times/miller_rabin_wheel" + std::to_string(w) + ".csv");
        std::ofstream res("results/resmiller_rabin_wheel" + std::to_string(w) + ".res");

        outfile << "first n primes,";
        for(uint64_t i = 0; i < 5; i++){
            outfile << i << ", ";
        }
        outfile << "\n";

        for(uint64_t i = 1; i < n; i++){
            outfile << "10^" << i << ",";
            for(uint64_t j = 0; j < 5; j++){
                //P(false positive) = (1/4)^k. chance of error in ENTIRE test should be 0.1%
                // total generated is 111111110
                //k = ln(0.001 * (1/111111110))/ln(1/4) = 18.34 rounds
                miller_rabin_wheel(uint64_t(pow(10, i)), 19, w, &t[i]);
                // for(auto x: miller_rabin_wheel(uint64_t(pow(10, i)), 19, w, &t[i])){res << x << "\n";}
                std::cout << "Time: " << t[i] << "us" << std::endl;
                outfile << t[i] << ",";
            }

            outfile << std::endl;
        }
    }
    return;
}

//works for candidate n where n < 341,550,071,728,321 (3.4 x 10^14)
bool miller_rabin_logic_optimized(uint64_t candidate, uint64_t s, uint64_t d){
    std::vector<uint64_t> bases = {2, 3, 5, 7, 11, 13, 17};
    for(uint64_t i = 0; i < 7; i++){
        uint64_t a = bases[i];
        uint64_t x = expmod(a, d, candidate);
        uint64_t y = 0;
        for(int64_t j = 0; j < s; j++){
            y = __uint128_t(x*x) % candidate;
            if(y == 1 && x != 1 && x != (candidate - 1)){
                return false;
            }
            x = y;
        }
        if(y != 1){
            return false;
        }
    }
    return true;
}

std::vector<uint64_t> miller_rabin_wheel_optimized(uint64_t size, uint64_t wheel_size, uint64_t* time){
    wheel_return_t wheeldata = generate_wheel_steps(wheel_size);
    std::vector<uint64_t> primes = wheeldata.primes;
    std::vector<uint64_t> wheel_steps = wheeldata.wheel_steps;
    uint64_t candidate = wheeldata.candidate;
    
    primes.reserve(size);
    size_t wheel_index = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    while(primes.size() < size){
        uint64_t s = __builtin_ctzll(candidate - 1);
        uint64_t d = (candidate - 1) >> s;

        if(miller_rabin_logic_optimized(candidate, s, d)){
            primes.emplace_back(candidate);
        }
                
        candidate += wheel_steps[wheel_index];
        wheel_index = (wheel_index + 1) % wheel_steps.size();
    }

    int64_t duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time).count();

    if(time){*time = duration;}
    return primes;
}

void test_miller_rabin_wheel_optimized(uint64_t n, uint64_t wheel_size){
    for(uint64_t w = 2; w < wheel_size; w++){
        std::vector<uint64_t> f(n);
        std::vector<uint64_t> t(n);
        std::ofstream outfile("times/miller_rabin_wheel" + std::to_string(w) + "_optimized.csv");
        std::ofstream res("results/resmiller_rabin_wheel" + std::to_string(w) + "_optimized.res");

        outfile << "first n primes,";
        for(uint64_t i = 0; i < 5; i++){
            outfile << i << ", ";
        }
        outfile << "\n";

        for(uint64_t i = 1; i < n; i++){
            outfile << "10^" << i << ",";
            for(uint64_t j = 0; j < 5; j++){
                miller_rabin_wheel_optimized(uint64_t(pow(10, i)), w, &t[i]);
                // for(auto x: miller_rabin_wheel_optimized(uint64_t(pow(10, i)), w, &t[i])){res << x  << "\n";}
                std::cout << "Time: " << t[i] << "us" << std::endl;
                outfile << t[i] << ","; 
            }

            outfile << std::endl;
        }
    }
    std::cout << "done" << std::endl;
    return;
}

int main(){
    uint64_t n = 9;
    uint64_t wheel_size = 9;
    
    test_trial_division_naive(n);
    test_miller_rabin_naive(n);
    test_trial_division_wheel(n, wheel_size);
    test_miller_rabin_wheel(n, wheel_size);
    test_miller_rabin_wheel_optimized(n, wheel_size);
    return 0;
}