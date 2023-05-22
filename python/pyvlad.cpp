#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "VLAD.h"

namespace py = pybind11;
using namespace std;
using namespace VLAD;

PYBIND11_MODULE(pyvlad, m) {

  py::class_<Vocabulary> vocab(m, "Vocabulary");
  vocab.def(py::init<const std::string &>()); // load from file

  vocab.def("size", &Vocabulary::size);
  vocab.def("__repr__", [](Vocabulary &self) {
    std::stringstream ss;
    ss << self;
    return ss.str();
  });
  vocab.def("__str__", [](Vocabulary &self) {
    std::stringstream ss;
    ss << self;
    return ss.str();
  });

  py::class_<Database> db(m, "Database");
  db.def(py::init<const std::string &>());
  db.def(
      "add",
      [](Database &self, py::array_t<uint8_t> &features) {
        auto buffer_info = features.request();
        cv::Mat mat(buffer_info.shape[0], buffer_info.shape[1], CV_8UC1,
                    buffer_info.ptr);
        auto entry_id = self.add(mat);
        return entry_id;
      },
      py::arg("features"));
  db.def(
      "query",
      [](Database &self, py::array_t<uint8_t> &features, int max_results,
         int max_id) {
        auto buffer_info = features.request();
        cv::Mat mat(buffer_info.shape[0], buffer_info.shape[1], CV_8UC1,
                    buffer_info.ptr);
        QueryResults results;
        self.query(mat, results, max_results, max_id);

        py::list py_results;
        for (const auto &result : results) {
          py_results.append(py::make_tuple(result.id, result.score));
        }
        return py_results;
      },
      py::arg("features"), py::arg("max_results") = 1, py::arg("max_id") = -1);
  db.def("size", &Database::size);
  db.def("__repr__", [](Database &self) {
    std::stringstream ss;
    ss << self;
    return ss.str();
  });
  db.def("__str__", [](Database &self) {
    std::stringstream ss;
    ss << self;
    return ss.str();
  });
}
