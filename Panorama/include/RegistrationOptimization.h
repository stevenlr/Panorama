#ifndef REGISTRATION_OPTIMIZATION_H_
#define REGISTRATION_OPTIMIZATION_H_

#include <opencv2/core/core.hpp>

cv::Mat_<double> registerImages(const cv::Mat &sceneImage, const cv::Mat &objectImage, const cv::Mat &initialHomography);

#endif
