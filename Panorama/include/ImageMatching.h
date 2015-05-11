#ifndef IMAGE_MATCHING_H_
#define IMAGE_MATCHING_H_

#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

struct ImageDescriptor {
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat featureDescriptor;
};

struct ImageMatchInfos {
	float avgDistance;
	float minDistance;
	std::vector<cv::DMatch> matches;
};

ImageMatchInfos matchImages(const ImageDescriptor &sceneDescriptor, const ImageDescriptor &objectDescriptor);
cv::Mat computeHomography(const ImageDescriptor &sceneDescriptor, const ImageDescriptor &objectDescriptor, const ImageMatchInfos &match);

#endif
