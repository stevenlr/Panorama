#include "Calibration.h"

#include <opencv2/stitching/detail/autocalib.hpp>

using namespace std;
using namespace cv;

void cameraPoseFromHomography(const Mat &H, Mat &pose)
{
	pose = Mat::eye(3, 4, CV_64F);

	float norm1 = (float) norm(H.col(0));  
	float norm2 = (float) norm(H.col(1));  
	float tnorm = (norm1 + norm2) / 2;

	Mat p1 = H.col(0);
	Mat p2 = pose.col(0);

	normalize(p1, p2);

	p1 = H.col(1);
	p2 = pose.col(1);

	normalize(p1, p2);

	p1 = pose.col(0);
	p2 = pose.col(1);

	Mat p3 = p1.cross(p2);
	Mat c2 = pose.col(2);

	p3.copyTo(c2);
	pose.col(3) = H.col(2) / tnorm;
}

void findFocalLength(const Mat &homography, vector<double> &focalLengths)
{
	double f0, f1;
	bool ok0, ok1;

	detail::focalsFromHomography(homography, f0, f1, ok0, ok1);

	if (ok0 && ok1) {
		focalLengths.push_back(sqrt(f0 * f1));
	}
}

double getMedianFocalLength(vector<double> &focalLengths)
{
	double focalLength;
	int numFocalLengths = focalLengths.size();

	std::sort(focalLengths.begin(), focalLengths.end());

	if (focalLengths.size() % 2 == 0) {
		focalLength = (focalLengths[numFocalLengths / 2 - 1] + focalLengths[numFocalLengths / 2]) / 2;
	} else {
		focalLength = focalLengths[numFocalLengths / 2];
	}

	return focalLength;
}

void findAnglesFromPose(const Mat &pose, double &rx, double &ry, double &rz)
{
	if (abs(pose.at<double>(2, 0)) != 1) {
		double y1 = -asin(pose.at<double>(2, 0));
		double y2 = PI - y1;

		double x1 = atan2(pose.at<double>(2, 1) / cos(y1), pose.at<double>(2, 2) / cos(y1));
		double x2 = atan2(pose.at<double>(2, 1) / cos(y2), pose.at<double>(2, 2) / cos(y2));

		double z1 = atan2(pose.at<double>(1, 0) / cos(y1), pose.at<double>(0, 0) / cos(y1));
		double z2 = atan2(pose.at<double>(1, 0) / cos(y2), pose.at<double>(0, 0) / cos(y2));

		if (abs(y1) < abs(y2)) {
			rx = x1;
			ry = y1;
			rz = z1;
		} else {
			rx = x2;
			ry = y2;
			rz = z2;
		}
	} else {
		rz = 0;

		if (pose.at<double>(2, 0) == -1) {
			ry = PI / 2;
			rz = atan2(pose.at<double>(0, 1), pose.at<double>(0, 2));
		} else {
			ry = -PI / 2;
			rz = atan2(-pose.at<double>(0, 1), -pose.at<double>(0, 2));
		}
	}
}
