#include "VLAD.h"
#include "test_utils.h"
#include <opencv2/opencv.hpp>
#include <set>
#include <vector>


void testVLADDDatabase() {

  PRINT_YELLOW("[Loading vocabulary] start");
  std::cout << "Current path is " << fs::current_path() << '\n';
  VLAD::Database db("../../config/sthereo_01_rgb_4_3.yaml");
  std::cout << db << std::endl;
  PRINT_GREEN("[Loading vocabulary] end\n");


  PRINT_YELLOW("[VLAD::Database::add] start");
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
  PRINT_GREEN("[VLAD::Database::add] end\n");


  PRINT_YELLOW("[VLAD::Database::query] start");
  for (auto &path : paths) {
    cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptor;
    orb->detectAndCompute(image, cv::Mat(), keypoints, descriptor);

    VLAD::QueryResults results;
    db.query(descriptor, results, 5);

    std::cout << "Query results: " << results << std::endl;
    break;
  }
  PRINT_GREEN("[VLAD::Database::query] end");
}


int main() {
  testVLADDDatabase();
  return 0;
}