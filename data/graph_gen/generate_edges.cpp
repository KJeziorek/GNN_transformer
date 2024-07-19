#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Function to calculate Euclidean distance between two points
double euclidean_distance(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return std::sqrt(sum);
}

// Function to generate edges
std::vector<std::pair<int, int>> generate_edges(const std::vector<std::vector<double>>& pos, double radius, int max_num_neighbors, int queue_size) {
    int num_nodes = pos.size();
    std::vector<std::pair<int, int>> edges;

    for (int i = 0; i < num_nodes; ++i) {
        std::vector<std::pair<double, int>> distances;

        int start = std::max(0, i-queue_size);
        for (int j = start; j <= i; ++j) {

            double dist = euclidean_distance(pos[i], pos[j]);

            if (dist < radius) {
                edges.emplace_back(j, i);
            }
        }
    }

    return edges;
}

// Binding code
namespace py = pybind11;

PYBIND11_MODULE(generate_edges, m) {
    m.def("generate_edges", &generate_edges, "Generate edges",
          py::arg("pos"), py::arg("radius"), py::arg("max_num_neighbors"), py::arg("queue_size"));
}
