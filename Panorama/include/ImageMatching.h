#ifndef IMAGE_MATCHING_H_
#define IMAGE_MATCHING_H_

#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

struct ImageDescriptor {
	int image;
	int width;
	int height;
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat featureDescriptor;
};

struct ImageMatchInfos {
	float avgDistance;
	float minDistance;
	std::vector<std::pair<int, int>> matches;
	cv::Mat homography;
	float confidence;

	ImageMatchInfos() = default;
	ImageMatchInfos(const ImageMatchInfos &infos);
	ImageMatchInfos &operator=(const ImageMatchInfos &infos);
};

typedef std::pair<std::pair<int, int>, float> MatchGraphEdge;

inline bool compareMatchGraphEdge(const MatchGraphEdge &first, const MatchGraphEdge &second)
{
	return first.second > second.second;
}

ImageMatchInfos matchImages(const ImageDescriptor &sceneDescriptor, const ImageDescriptor &objectDescriptor);
cv::Mat computeHomography(const ImageDescriptor &sceneDescriptor, const ImageDescriptor &objectDescriptor, ImageMatchInfos &match);

#endif
