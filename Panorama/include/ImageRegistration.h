#ifndef IMAGE_REGISTRATION_H_
#define IMAGE_REGISTRATION_H_

#include <vector>
#include <mutex>

#include <opencv2/core/core.hpp>

#include "ImagesRegistry.h"
#include "MatchGraph.h"

struct ComputeHomographyOutput {
	std::vector<uchar> inliersMask;
	int nbInliers;
	int nbOverlaps;
	cv::Mat homography;
};

void computeHomography(const ImageDescriptor &sceneDescriptor, const ImageDescriptor &objectDescriptor, const ImageMatchInfos &match, ComputeHomographyOutput &output);
bool matchImages(const ImageDescriptor &sceneDescriptor, const ImageDescriptor &objectDescriptor, ImageMatchInfos &matchInfos, std::mutex *matchMutex);

#endif
