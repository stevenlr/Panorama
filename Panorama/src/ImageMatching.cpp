#include "ImageMatching.h"

using namespace std;
using namespace cv;

ImageMatchInfos matchImages(const ImageDescriptor &sceneDescriptor, const ImageDescriptor &objectDescriptor)
{
	ImageMatchInfos matchInfos;
	Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create("FlannBased");

	descriptorMatcher->match(objectDescriptor.featureDescriptor, sceneDescriptor.featureDescriptor, matchInfos.matches);

	matchInfos.minDistance = numeric_limits<float>::max();
	matchInfos.avgDistance = 0;

	for (int i = 0; i < matchInfos.matches.size(); ++i) {
		float dist = matchInfos.matches[i].distance;

		matchInfos.avgDistance += dist;

		if (dist < matchInfos.minDistance) {
			matchInfos.minDistance = dist;
		}
	}

	matchInfos.avgDistance /= matchInfos.matches.size();

	return matchInfos;
}

Mat computeHomography(const ImageDescriptor &sceneDescriptor, const ImageDescriptor &objectDescriptor, const ImageMatchInfos &match)
{
	vector<DMatch> goodMatches;

	for (int i = 0; i < match.matches.size(); ++i) {
		if (match.matches[i].distance < 3 * match.minDistance) {
			goodMatches.push_back(match.matches[i]);
		}
	}

	vector<Point2f> points[2];

	for (int i = 0; i < goodMatches.size(); ++i) {
		points[0].push_back(sceneDescriptor.keypoints[goodMatches[i].trainIdx].pt);
		points[1].push_back(objectDescriptor.keypoints[goodMatches[i].queryIdx].pt);
	}

	return findHomography(points[1], points[0], CV_RANSAC);
}
