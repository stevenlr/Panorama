#ifndef CALIBRATION_H_
#define CALIBRATION_H_

#include <vector>

#include <opencv2/core/core.hpp>

void cameraPoseFromHomography(const cv::Mat &H, cv::Mat &pose);
void findFocalLength(const cv::Mat &homography, std::vector<double> &focalLengths);
double getMedianFocalLength(std::vector<double> &focalLengths);
void findAnglesFromPose(const cv::Mat &pose, double &rx, double &ry, double &rz);

#endif
