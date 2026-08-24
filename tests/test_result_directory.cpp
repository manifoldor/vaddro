#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

#include "result_io.hpp"

static bool is_directory(const std::string &path) {
    struct stat info {};
    return lstat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode);
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;

    const std::string root = argv[1];
    const std::string output = root + "/result";
    std::string error;

    if (!vaddro::ensure_output_directory(output, error) || !is_directory(output)) {
        std::cerr << error << '\n';
        return 1;
    }

    if (rmdir(output.c_str()) != 0) {
        std::cerr << "could not remove simulated live result tree\n";
        return 1;
    }

    error.clear();
    if (!vaddro::ensure_output_directory(output, error) || !is_directory(output)) {
        std::cerr << "deleted result tree was not restored: " << error << '\n';
        return 1;
    }

    struct stat info {};
    if (lstat(output.c_str(), &info) != 0 || (info.st_mode & 0777) != 0700) {
        std::cerr << "restored directory does not use mode 0700\n";
        return 1;
    }

    const std::string blocked_root = root + "/blocked";
    {
        std::ofstream blocked(blocked_root);
        blocked << "not a directory\n";
    }
    error.clear();
    if (vaddro::ensure_output_directory(blocked_root + "/run", error)) {
        std::cerr << "regular-file path component was accepted\n";
        return 1;
    }

    const std::string symlink_root = root + "/linked";
    if (symlink(output.c_str(), symlink_root.c_str()) != 0) {
        std::cerr << "could not create symlink test fixture\n";
        return 1;
    }
    error.clear();
    if (vaddro::ensure_output_directory(symlink_root + "/child", error)) {
        std::cerr << "symlink path component was accepted\n";
        return 1;
    }

    std::cout << "Result directory recovery test: OK\n";
    return 0;
}
