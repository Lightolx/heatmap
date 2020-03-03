//
// Created by lightol on 2019/11/29.
//

#include <iostream>
#include <vector>

#include <eigen3/Eigen/Eigen>
#include <opencv2/core.hpp>
#include <cv.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/transforms.h>

using PointT = pcl::PointXYZ;
using PointCloudT = pcl::PointCloud<PointT>;

using std::cout;
using std::endl;

int main() {
    // Step0: 数据读取，包括原图、2D分割结果、SemanticMap以及camera pose的ground truth
    cv::Mat image_raw = cv::imread("../data/009060.png");
    cv::Mat image_seg = cv::imread("../data/segmentation.png");

//    cv::imshow("ini", image_seg);
//    cv::waitKey();

    int rows = image_seg.rows;
    int cols = image_seg.cols;

    // 分离出所有属于箭头的像素
    cv::Mat image_arrow(image_seg.size(), CV_8UC1, cv::Scalar::all(0));
    std::vector<Eigen::Vector2d> vSeedPts;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (image_seg.at<cv::Vec3b>(i, j) == cv::Vec3b(55, 80, 156)) {
                image_arrow.at<uchar>(i, j) = 255;
                vSeedPts.emplace_back(i, j);
            }
        }
    }
//    cv::imshow("arrow", image_arrow);
//    cv::waitKey();

    Eigen::Matrix4d Tcw;
    Tcw << -0.811759,    0.583991,  -0.00150408, -90.8853,
            -0.00706828, -0.0124003, -0.999898,   13.4735,
            -0.58395,    -0.811666,   0.0141939,  -202.831,
            0,           0,          0,           1;

    PointCloudT::Ptr cloud(new PointCloudT);
    pcl::io::loadPLYFile("../data/arrow.ply", *cloud);

    // Step0.5: 重投影检查在ground truth下的效果
    PointCloudT::Ptr cloud_in_cam(new PointCloudT);
    pcl::transformPointCloud(*cloud, *cloud_in_cam, Tcw);
    for (const auto &pt : cloud_in_cam->points) {

    }

    // Step1: 对每个点都以其为中心建立一个正态分布，算出这个正态分布在图像上每个像素点的值
    cv::Mat image(rows, cols, CV_32FC1, cv::Scalar::all(0));
    cout << "vSeedPts.size = " << vSeedPts.size() << endl;
    for (int i = 0; i < rows; ++i) {
        cout << "i = " << i << endl;
        for (int j = 0; j < cols; ++j) {
            double sum_score = 0.0;
            for (const Eigen::Vector2d &pt: vSeedPts) {
//                continue;
                double dist = (Eigen::Vector2d(i, j) - pt).norm();
                if (dist > 1000) {
                    continue;
                }
                double score = 1 / (dist + 20.0);
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
    cout << "min = " << min_value << ", max = " << max_value << endl;
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