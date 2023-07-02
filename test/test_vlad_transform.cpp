#include "DBoW3.h"
#include "VLAD.h"
#include "test_utils.h"
#include <opencv2/opencv.hpp>


void testTrasformVLAD() {
  PRINT_YELLOW("testing VLAD::Vocabulary::transform");
  // load vocabulary
  DBoW3::Vocabulary dbow_vocab("../../config/sthereo_01_rgb_4_3.yaml");
  VLAD::Vocabulary vlad_vocab(std::make_shared<DBoW3::Vocabulary>(dbow_vocab));
  std::cout << vlad_vocab << std::endl;

  auto orb = cv::ORB::create();

  std::vector<std::string> paths = std::move(loadImagePaths("../../assets/01"));
  cv::Mat image = cv::imread(paths[0], cv::IMREAD_GRAYSCALE);
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptor;
  orb->detectAndCompute(image, cv::Mat(), keypoints, descriptor);

  VLAD::AggregationVector vlad = vlad_vocab.transform(descriptor);
  std::cout << vlad << std::endl;
  PRINT_GREEN("test VLAD::Vocabulary::transform passed");
}

void testScoreVLAD() {
  PRINT_YELLOW("testing VLAD::Vocabulary::score");
  // load vocabulary
  DBoW3::Vocabulary dbow_vocab("../../config/sthereo_01_rgb_4_3.yaml");
  VLAD::Vocabulary vlad_vocab(std::make_shared<DBoW3::Vocabulary>(dbow_vocab));
  std::cout << vlad_vocab << std::endl;

  auto orb = cv::ORB::create();

  VLAD::AggregationVector prev_vlad;
  std::vector<std::string> paths = std::move(loadImagePaths("../../assets/01"));
  for (auto &path : paths) {
    cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptor;
    orb->detectAndCompute(image, cv::Mat(), keypoints, descriptor);
    VLAD::AggregationVector vlad = vlad_vocab.transform(descriptor);
    std::cout << "score: " << vlad_vocab.score(prev_vlad, vlad) << std::endl;
    prev_vlad = vlad;
  }
  PRINT_GREEN("test VLAD::Vocabulary::score passed");
}

int main() {
  testTrasformVLAD();
  testScoreVLAD();
  return 0;
}