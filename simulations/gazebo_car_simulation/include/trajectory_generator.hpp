#ifndef TRAJECTORY_GENERATOR_HPP
#define TRAJECTORY_GENERATOR_HPP

#include <vector>
#include <string>

class TrajectoryGenerator {
public:
    std::vector<std::vector<double>> loadPath(const std::string& filename);
};

#endif // TRAJECTORY_GENERATOR_HPP
