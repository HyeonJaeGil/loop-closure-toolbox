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
#include "VladQueryResults.h"
#include "VladVocabulary.h"
#include <deque>

namespace VLAD {
/**
 * @brief The database contains the keyframes and a pointer to the vocabulary.
 *
 */
class Database {
public:
  /**
   * Database constructor
   *
   * @param vocabulary_ptr shared pointer to the vocabulary
   */
  Database(std::shared_ptr<DBoW3::Vocabulary> vocabulary_ptr);

  /**
   * @brief Database constructor
   *
   * @param vocabulary_path path to the vocabulary
   */
  Database(const std::string &vocabulary_path);

  /**
   * @brief Database copy constructor
   *
   * @param other
   */
  Database(const Database &other);

  /**
   * @brief assignment operator
   * 
  */
  Database &operator=(const Database &other) = delete;

  /**
   * @brief return the size of the database
   */
  std::size_t size() const { return database_.size(); }

  /**
   * @brief empty the database
   *
   */
  void clear() { database_.clear(); }

  /**
   * Add descriptors to the database and return the index
   *
   * @param descriptors
   * @return index of the input descriptor
   */
  unsigned int add(const cv::Mat &descriptors);

  /**
   * Add descriptors (one per row) to the database and return the index
   *
   * @param descriptors
   * @return index of the input descriptor
   */
  unsigned int add(const std::vector<cv::Mat> &descriptors);

  /**
   * Add aggregation to the database and return the index
   *
   * @param aggregation
   * @return index of the input aggregation
   */
  unsigned int add(const AggregationVector &aggregation);

  /**
   * @brief query the database with the input descriptors
   * @param[in] descriptors query descriptors, one row per descriptor
   * @param[out] results QueryResults
   * @param[in] max_results maximum number of results to return. <= 0 means all
   * @param[in] max_id maximum id of the results to return. < 0 means all
   */
  void query(const cv::Mat &descriptors, VLAD::QueryResults &results,
             int max_results = -1, int max_id = -1);

  /**
   * @brief query the database with the input descriptors
   * @param[in] descriptors query descriptors
   * @param[out] results QueryResults
   * @param[in] max_results maximum number of results to return. <= 0 means all
   * @param[in] max_id maximum id of the results to return. < 0 means all
   */
  void query(const std::vector<cv::Mat> &descriptors,
             VLAD::QueryResults &results, int max_results = -1,
             int max_id = -1);

  /**
   * @brief query the database with the input aggregation
   * @param[in] aggregation query aggregation
   * @param[out] results QueryResults
   * @param[in] max_results maximum number of results to return. <= 0 means all
   * @param[in] max_id maximum id of the results to return. < 0 means all
   */
  void query(const AggregationVector &aggregation, VLAD::QueryResults &results,
             int max_results = -1, int max_id = -1);

  /**
   * @brief compute pairwise scores between all the element in the database
   */
  cv::Mat computePairwiseDistance() const;

  /**
   * @brief print the database information to the output stream
   */
  friend std::ostream &operator<<(std::ostream &os, const Database &db) {
    os << db.vocabulary_ << std::endl;
    os << "Database size: " << db.database_.size() << std::endl;
    return os;
  }


private:
  Vocabulary vocabulary_;
  std::vector<AggregationVector> database_;
};

} // namespace VLAD