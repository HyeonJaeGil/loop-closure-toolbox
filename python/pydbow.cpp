#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "DBoW3.h"

namespace py = pybind11;
using namespace std;
using namespace DBoW3;

std::string printVocabularyInfo(const Vocabulary &vocab) {
  stringstream ss;
  ss << "Vocabulary: k = " << vocab.getBranchingFactor() << ", L = " << vocab.getDepthLevels()
     << ", Weighting = ";

  std::string weighting_type(vocab.getWeightingType() == TF_IDF ?
                                 "tf-idf" :
                                 vocab.getWeightingType() == TF ?
                                 "tf" :
                                 vocab.getWeightingType() == IDF ?
                                 "idf" :
                                 vocab.getWeightingType() == BINARY ? "binary" : "unknown");
  ss << weighting_type << ", Scoring = ";

  std::string scoring_type(vocab.getScoringType() == L1_NORM ?
                               "L1-norm" :
                               vocab.getScoringType() == L2_NORM ?
                               "L2-norm" :
                               vocab.getScoringType() == CHI_SQUARE ?
                               "Chi square distance" :
                               vocab.getScoringType() == KL ?
                               "KL-divergence" :
                               vocab.getScoringType() == BHATTACHARYYA ?
                               "Bhattacharyya coefficient" :
                               vocab.getScoringType() == DOT_PRODUCT ? "Dot product" : "unknown");
  ss << scoring_type;

  ss << ", Number of words = " << vocab.size();
  return ss.str();
}

std::string printDatabaseInfo(const Database &db) {
  stringstream ss;
  ss << "Database: Entries = " << db.size()
     << ", "
        "Using direct index = "
     << (db.usingDirectIndex() ? "yes" : "no");

  if (db.usingDirectIndex())
    ss << ", Direct index levels = " << db.getDirectIndexLevels();

  ss << ". " << *db.getVocabulary();
  return ss.str();
}

PYBIND11_MODULE(pydbow, m) {

  py::enum_<WeightingType>(m, "WeightingType")
      .value("TF_IDF", WeightingType::TF_IDF)
      .value("TF", WeightingType::TF)
      .value("IDF", WeightingType::IDF)
      .value("BINARY", WeightingType::BINARY);

  py::enum_<ScoringType>(m, "ScoringType")
      .value("L1_NORM", ScoringType::L1_NORM)
      .value("L2_NORM", ScoringType::L2_NORM)
      .value("CHI_SQUARE", ScoringType::CHI_SQUARE)
      .value("KL", ScoringType::KL)
      .value("BHATTACHARYYA", ScoringType::BHATTACHARYYA)
      .value("DOT_PRODUCT", ScoringType::DOT_PRODUCT);

  py::class_<Vocabulary> vocab(m, "Vocabulary");
  vocab.def(py::init<int, int>());            // k, L
  vocab.def(py::init<const std::string &>()); // load from file
  vocab.def(
      "create",
      [](Vocabulary &self, const py::list &list_of_ndarray) {
        std::vector<cv::Mat> features;
        for (const auto &feature : list_of_ndarray) {
          py::array_t<uint8_t> ndarray = feature.cast<py::array_t<uint8_t>>();
          auto buffer_info = ndarray.request();
          cv::Mat mat(buffer_info.shape[0], buffer_info.shape[1], CV_8UC1, buffer_info.ptr);
          features.push_back(mat);
        }
        self.create(features);
      },
      py::arg("training_features"));
  vocab.def(
      "save",
      [](Vocabulary &self, const std::string &filename, bool binary) {
        self.save(filename, binary);
      },
      py::arg("filename"), py::arg("binary") = true);
  vocab.def("size", &Vocabulary::size);
  vocab.def("__repr__", [](Vocabulary &self) { return printVocabularyInfo(self); });
  vocab.def("__str__", [](Vocabulary &self) { return printVocabularyInfo(self); });

  py::class_<Database> db(m, "Database");
  db.def(py::init<const Vocabulary &, bool, int>(), py::arg("voc"), py::arg("use_di") = true,
         py::arg("di_levels") = 0);
  db.def(
      "add",
      [](Database &self, py::array_t<uint8_t> &features) {
        auto buffer_info = features.request();
        cv::Mat mat(buffer_info.shape[0], buffer_info.shape[1], CV_8UC1, buffer_info.ptr);
        auto entry_id = self.add(mat);
        return entry_id;
      },
      py::arg("features"));
  db.def(
      "query",
      [](Database &self, py::array_t<uint8_t> &features, int max_results, int max_id) {
        auto buffer_info = features.request();
        cv::Mat mat(buffer_info.shape[0], buffer_info.shape[1], CV_8UC1, buffer_info.ptr);
        QueryResults results;
        self.query(mat, results, max_results, max_id);

        py::list py_results;
        for (const auto &result : results) {
          py_results.append(py::make_tuple(result.Id, result.Score));
        }
        return py_results;
      },
      py::arg("features"), py::arg("max_results") = 1, py::arg("max_id") = -1);
  db.def("size", &Database::size);
  db.def("__repr__", [](Database &self) { return printDatabaseInfo(self); });
  db.def("__str__", [](Database &self) { return printDatabaseInfo(self); });
}
