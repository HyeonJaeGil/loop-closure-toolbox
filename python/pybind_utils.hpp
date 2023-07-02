#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include <opencv2/opencv.hpp>

namespace py = pybind11;
using namespace std;

int toOpenCVType(const py::dtype &dtype) {
  if (dtype.is(py::dtype::of<uint8_t>()))
    return CV_8UC1;
  else if (dtype.is(py::dtype::of<uint16_t>()))
    return CV_16UC1;
  else if (dtype.is(py::dtype::of<float>()))
    return CV_32FC1;
  else if (dtype.is(py::dtype::of<double>()))
    return CV_64FC1;
  else
    throw std::runtime_error("Unsupported type passed to toMat");
}

template <typename T> cv::Mat toMat(const py::array_t<T> &input) {
  if (input.ndim() != 2)
    throw std::runtime_error("Number of dimensions must be two");
  int type = toOpenCVType(input.dtype());
  auto buf = input.request();
  cv::Mat mat(buf.shape[0], buf.shape[1], type, (void *)buf.ptr);
  return mat;
}

template <typename T> py::array_t<T> toArray(const cv::Mat &input) {
  // input cv::Mat -> py::buffer
  py::buffer_info buffer_info_descriptors =
      py::buffer_info(input.data, sizeof(T), py::format_descriptor<T>::format(), 2,
                      {input.rows, input.cols}, {input.step[0], input.step[1]});
  // py::buffer -> py::array
  return py::array_t<T>(buffer_info_descriptors);
}