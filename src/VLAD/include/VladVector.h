#pragma once
#include <opencv2/opencv.hpp>

namespace VLAD {

class AggregationVector {

public:
  explicit AggregationVector(const cv::Mat &aggregation = cv::Mat())
      : aggregation(aggregation) {}

  AggregationVector(const AggregationVector &agg)
      : aggregation(std::move(agg.aggregation)) {}

  AggregationVector &operator=(const AggregationVector &agg) {
    aggregation = agg.aggregation;
    return *this;
  }
  
  cv::Mat toMat() const { return aggregation; }

  bool empty() const { return aggregation.empty(); }

  // shape
  int rows() const { return aggregation.rows; }
  int cols() const { return aggregation.cols; }
  int channels() const { return aggregation.channels(); }
  int type() const { return aggregation.type(); }

  friend std::ostream &operator<<(std::ostream &os, const AggregationVector &agg) {
    os << "AggregationVector information: " << std::endl;
    os << " - rows: " << agg.rows() << std::endl;
    os << " - cols: " << agg.cols() << std::endl;
    os << " - channels: " << agg.channels() << std::endl;
    os << " - type: " << agg.type() << std::endl;
    return os;
  }

  cv::Mat aggregation;
};

} // namespace VLAD