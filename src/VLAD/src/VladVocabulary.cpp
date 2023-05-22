/*
 * Copyright 2020 California  Institute  of Technology (“Caltech”)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "VladVocabulary.h"
#include <utility>

namespace VLAD {

Vocabulary::Vocabulary(std::shared_ptr<DBoW3::Vocabulary> vocabulary)
    : vocabulary_(std::move(vocabulary)),
      d_length_(vocabulary_->getDescriptorSize()),
      clusters_n_(static_cast<int>(pow(vocabulary_->getBranchingFactor(),
                                       vocabulary_->getDepthLevels()))),
      v_length_(static_cast<int>(pow(vocabulary_->getBranchingFactor(),
                                     vocabulary_->getDepthLevels()) *
                                 vocabulary_->getDescriptorSize() * 8)) {}

Vocabulary::Vocabulary(const std::string &vocabulary_path)
    : vocabulary_(std::make_shared<DBoW3::Vocabulary>(vocabulary_path)),
      d_length_(vocabulary_->getDescriptorSize()),
      clusters_n_(static_cast<int>(pow(vocabulary_->getBranchingFactor(),
                                       vocabulary_->getDepthLevels()))),
      v_length_(static_cast<int>(pow(vocabulary_->getBranchingFactor(),
                                     vocabulary_->getDepthLevels()) *
                                 vocabulary_->getDescriptorSize() * 8)) {}

Vocabulary::Vocabulary(const Vocabulary &other)
    : vocabulary_(std::move(other.vocabulary_)), d_length_(other.d_length_),
      clusters_n_(other.clusters_n_), v_length_(other.v_length_) {}

// --------------------------------------------------------------------------

void Vocabulary::findCentroid(const cv::Mat &desc, cv::Mat &centroid,
                              unsigned int &id_centroid) {
  if (this->empty()){
    std::cerr << "Vocabulary is empty!" << std::endl;
    exit(1);
  }
  id_centroid = vocabulary_->transform(desc);
  vocabulary_->getWord(id_centroid).convertTo(centroid, CV_8UC1);
}

// --------------------------------------------------------------------------

AggregationVector Vocabulary::transform(const cv::Mat &features) {
  std::vector<cv::Mat> feature_vec(features.rows);
  for (int i = 0; i < features.rows; ++i) {
    feature_vec[i] = features.row(i);
  }
  return transform(feature_vec);
}

// --------------------------------------------------------------------------

AggregationVector Vocabulary::transform(const std::vector<cv::Mat> &features) {
  if (features.empty() || this->empty()) {
    return AggregationVector();
  }
  cv::Mat diff, sum;

  // Init an empty descriptor
  cv::Mat vlad = cv::Mat::zeros(clusters_n_, d_length_, CV_8UC1);

  // Find the closest centroid and compute the VLAD
  cv::Mat closest_centroid;
  unsigned int id_centroid;

  for (auto &feature : features) {
    findCentroid(feature, closest_centroid, id_centroid);

    closest_centroid.convertTo(closest_centroid, CV_8UC1);

    cv::bitwise_xor(feature, closest_centroid, diff);
    cv::bitwise_or(diff, vlad.row(static_cast<int>(id_centroid)),
                   vlad.row(static_cast<int>(id_centroid)));
  }

  // return the computed descriptor
  return AggregationVector(vlad);
}

// --------------------------------------------------------------------------

double Vocabulary::score(const AggregationVector &x,
                         const AggregationVector &y) const {

  cv::Mat x_mat = x.toMat();
  cv::Mat y_mat = y.toMat();
  if (x_mat.empty() || y_mat.empty()) {
    return 0.0;
  }
  cv::Mat res;
  cv::bitwise_xor(x_mat, y_mat, res);
  double norm = (v_length_ - cv::norm(res, cv::NORM_HAMMING)) / v_length_;
  return norm;
}

} // namespace VLAD