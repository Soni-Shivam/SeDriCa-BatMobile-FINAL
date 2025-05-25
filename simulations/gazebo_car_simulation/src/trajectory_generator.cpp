#include "trajectory_generator.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

std::vector<std::vector<double>> TrajectoryGenerator::loadPath(const std::string& filename) {
    std::vector<std::vector<double>> path;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return path;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        double x, y;
        char comma;
        if (ss >> x >> comma >> y) {
            path.push_back({x, y});
        }
    }
    file.close();
    return path;
}
