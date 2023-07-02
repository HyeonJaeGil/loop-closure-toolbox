#pragma once
#include <experimental/filesystem>
#include <set>
#include <vector>
#include <algorithm>

namespace fs = std::experimental::filesystem;

std::vector<std::string> loadImagePaths(const std::string &directory) {
  std::vector<std::string> images;
  for (auto &p : fs::directory_iterator(directory)) {
    if (p.path().extension() == ".png") {
      images.emplace_back(p.path().string());
    }
  }
  std::sort(images.begin(), images.end());
  return images;
}

#define PRINT_GREEN(x) std::cout << "\033[1;32m" << x << "\033[0m" << std::endl;

#define PRINT_RED(x) std::cout << "\033[1;31m" << x << "\033[0m" << std::endl;

#define PRINT_YELLOW(x) std::cout << "\033[1;33m" << x << "\033[0m" << std::endl;