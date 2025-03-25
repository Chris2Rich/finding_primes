#include <chrono>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>

auto t = std::chrono::nanoseconds(1000000000); //1 second
pid_t child_pid = -1;

void start_child() {
    child_pid = fork();
    if (child_pid == 0) {
        execl("./primes.exe", "Primes", nullptr);
        perror("execl failed");
        exit(EXIT_FAILURE);
    } else {
        perror("fork failed");
    }
}

void kill_child() {
    if (child_pid > 0) {
        kill(child_pid, SIGINT);
        waitpid(child_pid, nullptr, 0);
    }
}

int main(int argc, char** argv){
    start_child();
    auto start = std::chrono::high_resolution_clock::now();
    while(true){
        if(std::chrono::high_resolution_clock::now() - start > t){
            kill_child();
            break;
        }
    }
    return 0;
}