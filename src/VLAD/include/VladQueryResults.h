#pragma once
#include "DBoW3.h"
#include <vector>

namespace VLAD {

class Result {
public:

  Result() : id(0), score(0), inliers(0) {}

  Result(unsigned int id, double score, unsigned int inliers = 0)
      : id(id), score(score), inliers(inliers) {}
  
  Result(const DBoW3::Result &res) : id(res.Id), score(res.Score) {}
  
  inline bool operator<(const Result &r) const { return this->score < r.score; }

  inline bool operator>(const Result &r) const { return this->score > r.score; }

  inline bool operator==(const Result &r) const { return this->id == r.id; }

  inline bool operator!=(const Result &r) const { return this->id != r.id; }

  inline bool operator<=(const Result &r) const { return this->score <= r.score; }

  inline bool operator>=(const Result &r) const { return this->score >= r.score; }

  friend std::ostream &operator<<(std::ostream &os, const Result &ret);

  unsigned int id;
  double score;
  unsigned int inliers;
};

class QueryResults : public std::vector<Result> {
public:
  
  QueryResults() : std::vector<Result>() {}

  QueryResults(const std::vector<DBoW3::Result> &res) {
    for (auto &r : res) {
      this->emplace_back(r);
    }
  }

  inline void scaleScores(double factor) {
    for (auto &r : *this) {
      r.score *= factor;
    }
  }
  
  friend std::ostream &operator<<(std::ostream &os, const QueryResults &ret);


};

} // namespace VLAD