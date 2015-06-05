#include "Camera.h"

#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;

Camera::Camera() :
		focalLength(1), width(1), height(1)
{
	rotation = Mat::eye(Size(1, 3), CV_64F);
}

Mat_<double> Camera::getK() const
{
	Mat_<double> K = Mat::eye(Size(3, 3), CV_64F);

	K(0, 0) = focalLength;
	K(1, 1) = focalLength * getAspectRatio();
	K(0, 2) = ppx;
	K(1, 2) = ppy;

	return K;
}

double Camera::getAspectRatio() const
{
	return static_cast<double>(height) / width;
}

cv::Mat_<double> Camera::getH() const
{
	return getK() * getR();
}

cv::Mat_<double> Camera::getR() const
{
	Mat R;

	Rodrigues(rotation, R);

	if (determinant(R) < 0) {
		R *= -1;
	}

	return R;
}
