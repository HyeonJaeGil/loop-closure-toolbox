#include "DBoW3.h"
#include "VLAD.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind_utils.hpp"

namespace py = pybind11;

namespace loopclosuretoolbox {

PYBIND11_MODULE(loopclosuretoolbox, m) {
  // Import the submodules
  py::module m_dbow = m.def_submodule("dbow");
  py::module m_vlad = m.def_submodule("vlad");

  py::enum_<DBoW3::WeightingType>(m_dbow, "WeightingType")
      .value("TF_IDF", DBoW3::WeightingType::TF_IDF)
      .value("TF", DBoW3::WeightingType::TF)
      .value("IDF", DBoW3::WeightingType::IDF)
      .value("BINARY", DBoW3::WeightingType::BINARY);

  py::enum_<DBoW3::ScoringType>(m_dbow, "ScoringType")
      .value("L1_NORM", DBoW3::ScoringType::L1_NORM)
      .value("L2_NORM", DBoW3::ScoringType::L2_NORM)
      .value("CHI_SQUARE", DBoW3::ScoringType::CHI_SQUARE)
      .value("KL", DBoW3::ScoringType::KL)
      .value("BHATTACHARYYA", DBoW3::ScoringType::BHATTACHARYYA)
      .value("DOT_PRODUCT", DBoW3::ScoringType::DOT_PRODUCT);

  py::class_<DBoW3::Vocabulary> vocab_dbow(m_dbow, "Vocabulary");
  vocab_dbow.def(py::init<int, int>());            // k, L
  vocab_dbow.def(py::init<const std::string &>()); // load from file
  vocab_dbow.def(
      "create",
      [](DBoW3::Vocabulary &self, const py::list &list_of_ndarray) {
        std::vector<cv::Mat> features;
        for (const auto &feature : list_of_ndarray) {
          py::array_t<uint8_t> feature_casted = feature.cast<py::array_t<uint8_t>>();
          cv::Mat feature_mat = toMat<uint8_t>(feature_casted);
          features.push_back(feature_mat);
        }
        self.create(features);
      },
      py::arg("training_features"));
  vocab_dbow.def(
      "save",
      [](DBoW3::Vocabulary &self, const std::string &filename, bool binary) {
        self.save(filename, binary);
      },
      py::arg("filename"), py::arg("binary") = true);
  vocab_dbow.def("size", &DBoW3::Vocabulary::size);
  vocab_dbow.def("__repr__", [](DBoW3::Vocabulary &self) {
    std::stringstream ss;
    ss << self;
    return ss.str();
  });
  vocab_dbow.def("__str__", [](DBoW3::Vocabulary &self) {
    std::stringstream ss;
    ss << self;
    return ss.str();
  });

  py::class_<DBoW3::Database> db_dbow(m_dbow, "Database");
  db_dbow.def(py::init<const DBoW3::Vocabulary &, bool, int>(), py::arg("voc"),
              py::arg("use_di") = true, py::arg("di_levels") = 0);
  db_dbow.def(
      "add",
      [](DBoW3::Database &self, py::array_t<uint8_t> &features) {
        cv::Mat mat = toMat<uint8_t>(features);
        auto entry_id = self.add(mat);
        return entry_id;
      },
      py::arg("features"));
  db_dbow.def(
      "query",
      [](DBoW3::Database &self, py::array_t<uint8_t> &features, int max_results, int max_id) {
        cv::Mat mat = toMat<uint8_t>(features);
        DBoW3::QueryResults results;
        self.query(mat, results, max_results, max_id);

        py::list py_results;
        for (const auto &result : results) {
          py_results.append(py::make_tuple(result.Id, result.Score));
        }
        return py_results;
      },
      py::arg("features"), py::arg("max_results") = 1, py::arg("max_id") = -1);
  db_dbow.def("compute_pairwise_score", [](DBoW3::Database &self) {
    cv::Mat pscore_mat = self.computepairwiseScore();
    auto pscore_array = toArray<double>(pscore_mat);
    return pscore_array;
  });
  db_dbow.def("size", &DBoW3::Database::size);
  db_dbow.def("__repr__", [](DBoW3::Database &self) {
    std::stringstream ss;
    ss << self;
    return ss.str();
  });
  db_dbow.def("__str__", [](DBoW3::Database &self) {
    std::stringstream ss;
    ss << self;
    return ss.str();
  });

  py::class_<VLAD::Vocabulary> vocab_vlad(m_vlad, "Vocabulary");
  vocab_vlad.def(py::init<const std::string &>()); // load from file
  vocab_vlad.def("size", &VLAD::Vocabulary::size);
  vocab_vlad.def("__repr__", [](VLAD::Vocabulary &self) {
    std::stringstream ss;
    ss << self;
    return ss.str();
  });
  vocab_vlad.def("__str__", [](VLAD::Vocabulary &self) {
    std::stringstream ss;
    ss << self;
    return ss.str();
  });

  py::class_<VLAD::Database> db_vlad(m_vlad, "Database");
  db_vlad.def(py::init<const std::string &>());
  db_vlad.def(
      "add",
      [](VLAD::Database &self, py::array_t<uint8_t> &features) {
        cv::Mat mat = toMat<uint8_t>(features);
        auto entry_id = self.add(mat);
        return entry_id;
      },
      py::arg("features"));
  db_vlad.def(
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
  db_vlad.def("compute_pairwise_score", [](VLAD::Database &self) {
    cv::Mat pscore_mat = self.computepairwiseScore();
    auto pscore_array = toArray<double>(pscore_mat);
    return pscore_array;
  });
  db_vlad.def("size", &VLAD::Database::size);
  db_vlad.def("__repr__", [](VLAD::Database &self) {
    std::stringstream ss;
    ss << self;
    return ss.str();
  });
  db_vlad.def("__str__", [](VLAD::Database &self) {
    std::stringstream ss;
    ss << self;
    return ss.str();
  });
}

} // namespace loopclosuretoolbox