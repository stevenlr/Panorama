#ifndef CAMERA_H_
#define CAMERA_H_

#include <opencv2/core/core.hpp>

class Camera {
public:
	Camera();

	cv::Mat_<double> getK() const;
	cv::Mat_<double> getH() const;
	cv::Mat_<double> getR() const;
	double getAspectRatio() const;

	cv::Mat_<double> rotation;
	double focalLength;
	int width;
	int height;
	double ppx;
	double ppy;
};

#endif
