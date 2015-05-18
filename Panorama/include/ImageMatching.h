#ifndef IMAGE_MATCHING_H_
#define IMAGE_MATCHING_H_

#include <vector>
#include <map>
#include <list>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "Scene.h"

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

struct MatchGraphEdge {
	int objectImage;
	int sceneImage;
	float confidence;
};

inline bool compareMatchGraphEdge(const MatchGraphEdge &first, const MatchGraphEdge &second)
{
	return first.confidence > second.confidence;
}

ImageMatchInfos matchImages(const ImageDescriptor &sceneDescriptor, const ImageDescriptor &objectDescriptor);
cv::Mat computeHomography(const ImageDescriptor &sceneDescriptor, const ImageDescriptor &objectDescriptor, ImageMatchInfos &match);
void extractFeatures(const Scene &scene, std::vector<ImageDescriptor> &descriptors);
void pairwiseMatch(const Scene &scene,
				   const std::vector<ImageDescriptor> &descriptors,
				   std::list<MatchGraphEdge> &matchGraphEdges,
				   std::map<std::pair<int, int>, ImageMatchInfos> &matchInfosMap,
				   std::vector<double> &focalLengths);

#endif
