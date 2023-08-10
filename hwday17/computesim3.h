#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <chrono>
#include <sophus/se3.hpp>

using namespace std;
using namespace cv;

#define probability 0.99
#define minInliers 6
#define maxIterations 300

int mnInliersi;
std::vector<bool> mvbInliersi;
vector<KeyPoint> keypoints_1, keypoints_2;
std::vector<cv::Mat> pts1, pts2;
std::vector<cv::Mat> mvP1im1;
std::vector<cv::Mat> mvP2im2;

vector<size_t> vAvailableIndices;
std::vector<size_t> mvnIndices1;
cv::Mat mT12i;
cv::Mat mT21i;

std::vector<size_t> mvnMaxError1;           // 当前关键帧中的某个特征点所允许的最大不确定度(和所在的金字塔图层有关)
std::vector<size_t> mvnMaxError2;           // 闭环关键帧中的某个特征点所允许的最大不确定度(同上)

cv::Mat K = (Mat_<float>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

void find_feature_matches(
  const Mat &img_1, const Mat &img_2,
  std::vector<KeyPoint> &keypoints_1,
  std::vector<KeyPoint> &keypoints_2,
  std::vector<DMatch> &matches);

// 像素坐标转相机归一化坐标
Point2d pixel2cam(const Point2d &p);

void pose_estimation_3d3d(
  const vector<Point3f> &pts1,
  const vector<Point3f> &pts2,
  Mat &R, Mat &t
);

void bundleAdjustment(
  const vector<Point3f> &points_3d,
  const vector<Point3f> &points_2d,
  Mat &R, Mat &t
);

/**
 * @brief 给出三个点,计算它们的质心以及去质心之后的坐标
 * 
 * @param[in] P     输入的3D点
 * @param[in] Pr    去质心后的点
 * @param[in] C     质心
 */
void ComputeCentroid(cv::Mat &P, cv::Mat &Pr, cv::Mat &C);


/**
 * @brief 按照给定的Sim3变换进行投影操作,得到三维点的2D投影点
 * 
 * @param[in] vP3Dw         3D点
 * @param[in & out] vP2D    投影到图像的2D点
 * @param[in] Tcw           Sim3变换
 */
void Project(vector<cv::Mat> &vP3Dw, vector<cv::Mat> &vP2D, cv::Mat Tcw);

/**
 * @brief 通过计算的Sim3投影，和自身投影的误差比较，进行内点检测
 * 
 */
void CheckInliers();

/**
 * @brief 根据两组匹配的3D点,计算P2到P1的Sim3变换
 * @param[in] pts1                  匹配的3D点(三个,每个的坐标都是列向量形式,三个点组成了3x3的矩阵)(当前关键帧)
 * @param[in] Pts2                  匹配的3D点(闭环关键帧)
 * @param[in] vAvailableIndices     匹配点对id
 * @return cv::Mat                  计算得到的Sim3矩阵
 */
cv::Mat ComputeSim3(vector<cv::Mat> &pts1, vector<cv::Mat> &pts2, vector<size_t> &vAvailableIndices);

float ComputerSigmaSquare(cv::KeyPoint keypoints);