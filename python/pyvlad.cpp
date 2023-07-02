#include "VLAD.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind_utils.hpp"

namespace py = pybind11;
using namespace std;
using namespace VLAD;

PYBIND11_MODULE(pyvlad, m) {

  py::class_<VLAD::Vocabulary> vocab(m, "Vocabulary");
  vocab.def(py::init<const std::string &>()); // load from file
  vocab.def("size", &VLAD::Vocabulary::size);
  vocab.def("__repr__", [](VLAD::Vocabulary &self) {
    std::stringstream ss;
    ss << self;
    return ss.str();
  });
  vocab.def("__str__", [](VLAD::Vocabulary &self) {
    std::stringstream ss;
    ss << self;
    return ss.str();
  });

  py::class_<VLAD::Database> db(m, "Database");
  db.def(py::init<const std::string &>());
  db.def(
      "add",
      [](VLAD::Database &self, py::array_t<uint8_t> &features) {
        cv::Mat mat = toMat<uint8_t>(features);
        auto entry_id = self.add(mat);
        return entry_id;
      },
      py::arg("features"));
  db.def(
      "query",
      [](VLAD::Database &self, py::array_t<uint8_t> &features, int max_results, int max_id) {
        cv::Mat mat = toMat<uint8_t>(features);
        VLAD::QueryResults results;
        self.query(mat, results, max_results, max_id);

        py::list py_results;
        for (const auto &result : results) {
          py_results.append(py::make_tuple(result.id, result.score));
        }
        return py_results;
      },
      py::arg("features"), py::arg("max_results") = 1, py::arg("max_id") = -1);
  db.def("compute_pairwise_score", [](VLAD::Database &self) {
    cv::Mat pscore_mat = self.computepairwiseScore();
    auto pscore_array = toArray<double>(pscore_mat);
    return pscore_array;
  });
  db.def("size", &VLAD::Database::size);
  db.def("__repr__", [](VLAD::Database &self) {
    std::stringstream ss;
    ss << self;
    return ss.str();
  });
  db.def("__str__", [](VLAD::Database &self) {
    std::stringstream ss;
    ss << self;
    return ss.str();
  });
}
