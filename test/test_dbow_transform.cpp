#include "DBoW3.h"
#include "test_utils.h"
#include <opencv2/opencv.hpp>
#include <set>
#include <vector>

void testDBoWTransform() {

  PRINT_YELLOW("[Loading vocabulary] start");
  std::cout << "Current path is " << fs::current_path() << '\n';
  DBoW3::Vocabulary voc("../../config/sthereo_01_rgb_4_3.yaml");
  DBoW3::Database db(voc, false, 0); // false: do not use direct index (default)
  std::cout << db << std::endl;
  PRINT_GREEN("[Loading vocabulary] end\n");


  PRINT_YELLOW("[Vocabulary::m_words print] start");
  std::cout << "voc.m_words.size(): " << voc.size() << std::endl;
  for (unsigned int i = 0 ; i < voc.size() ; i++) {
    cv::Mat mat = voc.getWord(i);
    DBoW3::WordValue word_weight = voc.getWordWeight(i);
    std::cout << "word_id: " << i << " word_weight: " << word_weight << " mat: " << mat << std::endl;
  }
  PRINT_GREEN("[Vocabulary::m_words print] end\n");


  PRINT_YELLOW("[DBoW3::Vocabulary::transform] start");
  auto orb = cv::ORB::create();
  std::vector<std::string> paths = std::move(loadImagePaths("../../assets/01"));
  for (auto &path : paths) {
    cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptor;
    orb->detectAndCompute(image, cv::Mat(), keypoints, descriptor);
    std::cout << "number of keypoints: " << keypoints.size() << std::endl;

    DBoW3::BowVector bow_vec;
    voc.transform(descriptor, bow_vec);
    // print bow_vec
    for (auto &item : bow_vec) {
      std::cout << item.first << " " << item.second << std::endl;
    }
    break;
  }
  PRINT_GREEN("[DBoW3::Vocabulary::transform] end\n");
}


int main() {
  testDBoWTransform();
  return 0;
}