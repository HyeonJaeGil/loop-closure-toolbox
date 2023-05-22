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

#include "VladDatabase.h"

namespace VLAD {

Database::Database(std::shared_ptr<DBoW3::Vocabulary> vocabulary_ptr)
    : vocabulary_(std::move(vocabulary_ptr)) {}

Database::Database(const std::string &vocabulary_path)
    : vocabulary_(std::make_shared<DBoW3::Vocabulary>(vocabulary_path)) {}

Database::Database(const Database &other)
    : vocabulary_(other.vocabulary_), database_(std::move(other.database_)) {}

// --------------------------------------------------------------------------

unsigned int Database::add(const cv::Mat &descriptors) {
  AggregationVector aggregation = vocabulary_.transform(descriptors);
  return add(aggregation);
}

unsigned int Database::add(const std::vector<cv::Mat> &descriptors) {
  AggregationVector aggregation = vocabulary_.transform(descriptors);
  return add(aggregation);
}

unsigned int Database::add(const AggregationVector &aggregation) {
  if (aggregation.empty()) {
    std::cerr << "Empty aggregation is being inserted!" << std::endl;
  }
  database_.push_back(aggregation);
  if (database_.size() == 0) {
    std::cerr << "Database size is 0!, something is wrong!" << std::endl;
    exit(1);
  }
  return database_.size() - 1;
}

// --------------------------------------------------------------------------

void Database::query(const cv::Mat &descriptors, QueryResults &results,
                     int max_results, int max_id) {
  AggregationVector aggregation = vocabulary_.transform(descriptors);
  return query(aggregation, results, max_results, max_id);
}

void Database::query(const std::vector<cv::Mat> &descriptors,
                     QueryResults &results, int max_results, int max_id) {
  AggregationVector aggregation = vocabulary_.transform(descriptors);
  return query(aggregation, results, max_results, max_id);
}

void Database::query(const AggregationVector &aggregation, QueryResults &results,
                     int max_results, int max_id) {
  std::vector<std::pair<double, unsigned int>> scores;
  int idx = 0;
  for (auto &data : database_) {
    double score_tmp = vocabulary_.score(aggregation, data);
    if ((max_id < 0 || idx < max_id))
      scores.push_back(std::make_pair(score_tmp, idx));
    ++idx;
  }

  if (scores.size() == 0) {
    std::cerr << "No candidate found!" << std::endl;
    return;
  }

  // sort scores in descending order
  std::sort(scores.begin(), scores.end(),
            [](const std::pair<double, unsigned int> &left,
               const std::pair<double, unsigned int> &right) {
              return left.first > right.first;
            });

  // cut scores to max_results
  if (max_results > 0 && scores.size() > static_cast<std::size_t>(max_results))
    scores.resize(max_results);

  // copy scores to results
  results.resize(scores.size());
  for (std::size_t i = 0; i < scores.size(); ++i) {
    results[i].score = scores[i].first;
    results[i].id = scores[i].second;
  }
}

// --------------------------------------------------------------------------

cv::Mat Database::computePairwiseDistance() {
  int n = database_.size();
  if (n < 1) {
    std::cerr << "Database is empty, pair-wise distance cannot be computed!"
              << std::endl;
    return cv::Mat();
  }
  std::cout << "Computing pair-wise distance for " << n << " images"
            << std::endl;
  cv::Mat distances(n, n, CV_64F);

#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    for (int j = i; j < n; j++) {
      double dist = vocabulary_.score(database_[i], database_[j]);
      distances.at<double>(i, j) = dist;
      distances.at<double>(j, i) = dist;
    }
  }

  return distances;
}

} // namespace VLAD