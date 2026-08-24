#pragma once

#include <cerrno>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>

namespace vaddro {

inline bool ensure_output_directory(const std::string &path, std::string &error) {
    if (path.empty()) return true;

    std::string current;
    std::size_t offset = 0;
    if (path.front() == '/') {
        current = "/";
        offset = 1;
    }

    while (offset <= path.size()) {
        const std::size_t separator = path.find('/', offset);
        const std::string component = path.substr(
            offset,
            separator == std::string::npos ? std::string::npos : separator - offset
        );

        if (!component.empty() && component != ".") {
            if (component == "..") {
                error = "output directory must not contain '..': '" + path + "'";
                return false;
            }
            if (!current.empty() && current.back() != '/') current += '/';
            current += component;

            struct stat info {};
            if (lstat(current.c_str(), &info) == 0) {
                if (!S_ISDIR(info.st_mode)) {
                    error = "output path component is not a directory: '" + current + "'";
                    return false;
                }
            } else if (errno == ENOENT) {
                if (mkdir(current.c_str(), 0700) != 0) {
                    const int mkdir_error = errno;
                    if (mkdir_error != EEXIST || lstat(current.c_str(), &info) != 0 ||
                        !S_ISDIR(info.st_mode)) {
                        error = "could not create output directory '" + current + "': " +
                                std::strerror(mkdir_error);
                        return false;
                    }
                }
            } else {
                const int stat_error = errno;
                error = "could not inspect output directory '" + current + "': " +
                        std::strerror(stat_error);
                return false;
            }
        }

        if (separator == std::string::npos) break;
        offset = separator + 1;
    }

    return true;
}

}  // namespace vaddro
