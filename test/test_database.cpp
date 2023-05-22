#include "DBoW3.h"
#include "VLAD.h"
#include <experimental/filesystem>
#include <opencv2/opencv.hpp>
#include <set>
#include <vector>

namespace fs = std::experimental::filesystem;

std::vector<std::string> loadImagePaths(const std::string &directory) {
  std::vector<std::string> images;
  for (auto &p : fs::directory_iterator(directory)) {
    if (p.path().extension() == ".png") {
      images.emplace_back(p.path().string());
    }
  }
  std::sort(images.begin(), images.end());
  return images;
}

VLAD::Database addToDatabase() {
  auto orb = cv::ORB::create();

  std::cout << "Current path is " << fs::current_path() << '\n';

  VLAD::Database db("vocabulary.yml.gz");

  std::cout << db << std::endl;

  std::vector<std::string> paths = std::move(
      loadImagePaths("/home/hj/Dropbox/Dataset/STHEREO-raw/01/image/rgb_left"));

  for (auto &path : paths) {
    cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptor;
    orb->detectAndCompute(image, cv::Mat(), keypoints, descriptor);

    auto db_size = db.add(descriptor) + 1;
    std::cout << "Database size: " << db_size << "\r";

    if (db_size > 200)
      break;
  }
  std::cout << std::endl;
  std::cout << db << std::endl;
  return db;
}

void queryDatabase(VLAD::Database &db) {
  auto orb = cv::ORB::create();

  std::vector<std::string> paths = std::move(
      loadImagePaths("/home/hj/Dropbox/Dataset/STHEREO-raw/01/image/rgb_left"));

  for (auto &path : paths) {
    cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptor;
    orb->detectAndCompute(image, cv::Mat(), keypoints, descriptor);

    VLAD::QueryResults results;
    db.query(descriptor, results, 10);

    std::cout << "Query results: " << results << std::endl;
    break;
  }
}

void getPairwiseDistance(VLAD::Database &db){
    cv::Mat pdist = db.computePairwiseDistance();
    cv::imshow("pdist", pdist);
    cv::waitKey(0);
}


int main() {
  VLAD::Database db = addToDatabase();
  getPairwiseDistance(db);
  queryDatabase(db);

  return 0;
}