#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <stdint.h>
#include <tuple>

class DataGenerator {
public:
  DataGenerator(const std::tuple<int32_t, int32_t> &x_bounds,
                const std::tuple<int32_t, int32_t> &y_bounds, int32_t num_aps,
                int32_t num_examples)
      : x_bounds(x_bounds), y_bounds(y_bounds), num_aps(num_aps),
        num_examples(num_examples),
        x_dist(std::get<0>(x_bounds), std::get<1>(x_bounds)),
        y_dist(std::get<0>(y_bounds), std::get<1>(y_bounds)), norm(0, 1e-8) {}

  void generateData() {
    for (int32_t i = 0; i < num_aps; i++) {
      auto pair = std::make_tuple(x_dist(rng), y_dist(rng), geom_dist(rng) + 1);
      ap_locations.push_back(pair);
    }

    for (int32_t i = 0; i < num_examples; i++) {
      int32_t x_coord = x_dist(rng);
      int32_t y_coord = y_dist(rng);
      std::vector<int32_t> rssi;

      for (int32_t k = 0; k < num_aps; k++) {
        double l2_norm =
            std::sqrt(std::pow(x_coord - std::get<0>(ap_locations[k]), 2) +
                      std::pow(y_coord - std::get<1>(ap_locations[k]), 2));
        double strength = 1 / std::pow(l2_norm, 2) + norm(rng);
        std::cout << strength << " ";
        double scaled = strength / (std::get<2>(ap_locations[k]) / 1000.);
        double db = 10 * std::log(scaled);
        rssi.push_back(db);
      }
      examples.push_back(std::make_tuple(x_coord, y_coord, rssi));
    }
  }

  std::vector<std::tuple<int32_t, int32_t, int32_t> > getAPLocations() {
    return ap_locations;
  }

  std::vector<std::tuple<int32_t, int32_t, std::vector<int32_t> > >
  getExamples() {
    return examples;
  }

private:
  std::tuple<int32_t, int32_t> x_bounds;
  std::tuple<int32_t, int32_t> y_bounds;
  int32_t num_aps;
  int32_t num_examples;
  std::uniform_int_distribution<int32_t> x_dist;
  std::uniform_int_distribution<int32_t> y_dist;
  std::default_random_engine rng;
  std::geometric_distribution<int32_t> geom_dist;
  std::normal_distribution<double> norm;

  std::vector<std::tuple<int32_t, int32_t, int32_t> > ap_locations;
  std::vector<std::tuple<int32_t, int32_t, std::vector<int32_t> > > examples;
};

int main(int argc, char **argv) {
  DataGenerator g(std::make_tuple(0, 1024), std::make_tuple(0, 4096), 100,
                  1000);
  g.generateData();
  auto examples = g.getExamples();
  std::ofstream of("out.txt");
  of << "x,y,";
  for (int i=0; i<100;i++) {
    of << i << ",";
  }
  of << std::endl;
  for (auto x : examples) {
    of << std::get<0>(x) << "," << std::get<1>(x) << ",";
    for (const auto y : std::get<2>(x)) {
      of << y << ",";
    }
    of << std::endl;
  }
}
