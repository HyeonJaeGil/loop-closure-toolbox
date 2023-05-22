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

#pragma once
#include "DBoW3.h"
#include "VladVector.h"
#include <memory>

namespace VLAD {

class Vocabulary {
public:
  /**
   * Constructor
   *
   * @param vocabulary
   */
  explicit Vocabulary(std::shared_ptr<DBoW3::Vocabulary> vocabulary);

  /**
   * @brief Constructor from vocabulary path
   *
   * @param vocabulary_path
   */
  explicit Vocabulary(const std::string &vocabulary_path);

  /**
   * @brief Copy constructor
   *
   */
  Vocabulary(const Vocabulary &other);

  /**
   * @brief assignment operator (deleted)
   *
   */
  Vocabulary &operator=(const Vocabulary &other) = delete;

  /**
   * @brief transform a set of descriptors into a AggregationVector vector
   * @param features
   * @return AggregationVector vector
   *
   * @return the computed binary VLAD descriptor
   */
  [[nodiscard]] AggregationVector
  transform(const std::vector<cv::Mat> &features);

  /**
   * @brief transform a set of descriptors into a VLAD vector
   * @param features one row per descriptor
   * @return AggregationVector vector
   *
   */
  [[nodiscard]] AggregationVector transform(const cv::Mat &features);

  /**
   * Compute the similarity score between two VLADs
   * @param x VLAD
   * @param y VLAD
   * @return computed score
   */
  [[nodiscard]] double score(const AggregationVector &x,
                             const AggregationVector &y) const;

  /**
   * @brief Returns the number of words in the vocabulary
   * 
   * @return unsigned int 
   */
  unsigned int size() const { return vocabulary_->size(); }

  /**
   * @brief Returns true if the vocabulary is empty
   * 
   * @return bool 
   */
  bool empty() const { return vocabulary_->empty(); }

  /**
   * @brief Returns the branching factor of the tree
   * 
   * @return int 
   */
  int getBranchingFactor() const { return vocabulary_->getBranchingFactor(); }

  /**
   * @brief Returns the depth levels of the tree
   * 
   * @return int 
   */
  int getDepthLevels() const { return vocabulary_->getDepthLevels(); }

  /**
   * @brief Returns the descriptor type
   * 
   * @return WeightingType (TF_IDF, TF, IDF, BINARY)
   */
  DBoW3::WeightingType getWeightingType() const { return vocabulary_->getWeightingType(); }

  /**
   * @brief Returns the scoring type
   * 
   * @return ScoringType (L1_NORM, L2_NORM, CHI_SQUARE, KL, BHATTACHARYYA, DOT_PRODUCT)
   */
  DBoW3::ScoringType getScoringType() const { return vocabulary_->getScoringType(); }

  /**
   * @brief Returns the descriptor size
   * 
   * @return int 
   */
  int getDescriptorSize() const { return vocabulary_->getDescriptorSize(); }


  /**
   * @brief print the vocabulary information
   *
   * @param os
   * @param voc
   * @return std::ostream&
   */
  friend std::ostream &operator<<(std::ostream &os, const Vocabulary &voc) {
    os << *voc.vocabulary_;
    return os;
  }

private:
  /**
   * Retrieve closest centroid to the selected descriptor
   *
   * @param[in] desc
   * @param[out] centroid
   * @param[out] id_centroid
   */
  void findCentroid(const cv::Mat &desc, cv::Mat &centroid,
                    unsigned int &id_centroid);

  std::shared_ptr<DBoW3::Vocabulary> vocabulary_;

  const int d_length_;
  const int clusters_n_;
  const int v_length_;
};

} // namespace VLAD