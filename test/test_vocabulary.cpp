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

void createVocabulary() {
  auto orb = cv::ORB::create();

  std::vector<std::string> paths = std::move(
      loadImagePaths("/home/hj/Dropbox/Dataset/STHEREO-raw/01/image/rgb_left"));

  std::vector<cv::Mat> descriptors;
  for (auto &path : paths) {
    cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptor;
    orb->detectAndCompute(image, cv::Mat(), keypoints, descriptor);
    descriptors.emplace_back(descriptor);

    // draw keypoints and imshow
    cv::Mat image_keypoints;
    cv::drawKeypoints(image, keypoints, image_keypoints);
    cv::imshow("Keypoints", image_keypoints);
    if (cv::waitKey(1) == 27 || descriptors.size() >= 2000) {
      break;
    }
  }

  // create vocabulary
  DBoW3::Vocabulary vocab(4, 3, DBoW3::TF_IDF, DBoW3::L1_NORM);
  vocab.create(descriptors);
  std::cout << "Vocabulary information: " << vocab << std::endl;
  vocab.save("vocabulary.yml.gz");
}

void transformVLAD() {

  // load vocabulary
  DBoW3::Vocabulary dbow_vocab("vocabulary.yml.gz");

  VLAD::Vocabulary vlad_vocab(std::make_shared<DBoW3::Vocabulary>(dbow_vocab));
  std::cout << vlad_vocab << std::endl;

  auto orb = cv::ORB::create();

  std::vector<std::string> paths = std::move(
      loadImagePaths("/home/hj/Dropbox/Dataset/STHEREO-raw/01/image/rgb_left"));

  for (auto &path : paths) {
    cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptor;
    orb->detectAndCompute(image, cv::Mat(), keypoints, descriptor);

    VLAD::AggregationVector vlad = vlad_vocab.transform(descriptor);
    std::cout << vlad << std::endl;

    break;
  }
}

void scoreVLAD() {

  // load vocabulary
  DBoW3::Vocabulary dbow_vocab("vocabulary.yml.gz");

  VLAD::Vocabulary vlad_vocab(std::make_shared<DBoW3::Vocabulary>(dbow_vocab));
  std::cout << vlad_vocab << std::endl;

  auto orb = cv::ORB::create();

  std::vector<std::string> paths = std::move(
      loadImagePaths("/home/hj/Dropbox/Dataset/STHEREO-raw/01/image/rgb_left"));

  VLAD::AggregationVector prev_vlad;
  for (auto &path : paths) {
    cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptor;
    orb->detectAndCompute(image, cv::Mat(), keypoints, descriptor);

    VLAD::AggregationVector vlad = vlad_vocab.transform(descriptor);
    std::cout << "score: " << vlad_vocab.score(prev_vlad, vlad) << std::endl;
    prev_vlad = vlad;
  }
}

int main() { scoreVLAD(); }