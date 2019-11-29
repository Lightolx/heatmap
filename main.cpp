#include <iostream>
#include <vector>

#include <eigen3/Eigen/Eigen>
#include <opencv2/core.hpp>
#include <cv.hpp>

using std::cout;
using std::endl;

int main() {
    int rows = 960;
    int cols = 1280;

    /*
    {
        cv::Mat image = cv::imread("/home/lightol/002554_mask.png");
        cv::imwrite("/home/lightol/002555.png", image*100);
        cv::imshow("image", image*100);
        cv::waitKey();
    }
     */

    // Step0: 首先生成一些2D上的散点
    std::default_random_engine engine;
    engine.seed(time(0));
    std::uniform_int_distribution<> uniformIntDistribution1(20, rows-20);
    std::uniform_int_distribution<> uniformIntDistribution2(20, cols-20);
    std::vector<Eigen::Vector2d> vSeedPts;
    for (int i = 0; i < 12; ++i) {
        vSeedPts.emplace_back(uniformIntDistribution1(engine), uniformIntDistribution2(engine));
    }

    cv::Mat image0(rows, cols, CV_8UC1, cv::Scalar::all(0));
    for (const auto &pt: vSeedPts) {
        cv::circle(image0, cv::Point(pt.y(), pt.x()), 5, cv::Scalar::all(255));
    }
    cv::imshow("sample image", image0);
//    cv::waitKey();
    // Step1: 对每个点都以其为中心建立一个正态分布，算出这个正态分布在图像上每个像素点的值
    cv::Mat image(rows, cols, CV_32FC1, cv::Scalar::all(0));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double sum_score = 0.0;
            for (const Eigen::Vector2d &pt: vSeedPts) {
                double score = 1 / ((Eigen::Vector2d(i, j) - pt).norm() + 20.0);
                sum_score += score;
            }

            image.at<float>(i, j) = sum_score;
        }
    }
//    cv::namedWindow("image", CV_WINDOW_FULLSCREEN);
//    cv::imshow("image", image);
    double min_value;
    double max_value;
    cv::minMaxIdx(image, &min_value, &max_value);
//    cv::imwrite("/home/lightol/Desktop/image.png", image);

    cv::Mat image_gray(image.size(), CV_8UC1);
//    cv::Mat image_gray;
    cv::convertScaleAbs(image, image_gray, 255 / max_value);
//    cv::normalize(image, image_gray, 0, 255, cv::NORM_MINMAX);
    cv::imshow("gray", image_gray);

    cv::Mat heatmap;
    cv::applyColorMap(image_gray, heatmap, cv::COLORMAP_JET);
    cv::imshow("heatmap image", heatmap);
    cv::waitKey();
}