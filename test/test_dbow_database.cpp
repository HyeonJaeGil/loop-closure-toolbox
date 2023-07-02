#include "DBoW3.h"
#include "test_utils.h"
#include <opencv2/opencv.hpp>
#include <set>
#include <vector>

void testDBowDatabase() {

  PRINT_YELLOW("[Loading vocabulary] start");
  std::cout << "Current path is " << fs::current_path() << '\n';
  DBoW3::Vocabulary voc("../../config/sthereo_07_rgb_4_3.yaml");
  DBoW3::Database db(voc, false, 0); // false: do not use direct index (default)
  std::cout << db << std::endl;
  PRINT_GREEN("[Loading vocabulary] end\n");


  PRINT_YELLOW("[DBoW3::Database::add] start");
  auto orb = cv::ORB::create();
  std::vector<std::string> paths = std::move(loadImagePaths("../../assets/01"));
  for (auto &path : paths) {
    cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptor;
    orb->detectAndCompute(image, cv::Mat(), keypoints, descriptor);
    auto db_size = db.add(descriptor) + 1;
  }
  std::cout << db << std::endl;
  PRINT_GREEN("[DBoW3::Database::add] end\n");


  PRINT_YELLOW("[DBoW3::Database::query] start");
  for (auto &path : paths) {
    cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptor;
    orb->detectAndCompute(image, cv::Mat(), keypoints, descriptor);

    DBoW3::QueryResults results;
    db.query(descriptor, results, 5);
    std::cout << "Query results: " << results << std::endl;
    break;
  }
  PRINT_GREEN("[DBoW3::Database::query] end");
}


int main() {
  testDBowDatabase();
  return 0;
}