#include "ImageMatching.h"

using namespace std;
using namespace cv;
#include <iostream>

ImageMatchInfos::ImageMatchInfos(const ImageMatchInfos &infos)
{
	avgDistance = infos.avgDistance;
	minDistance = infos.minDistance;
	confidence = infos.confidence;

	homography = infos.homography.clone();

	matches.resize(infos.matches.size());
	copy(infos.matches.begin(), infos.matches.end(), matches.begin());
}

ImageMatchInfos &ImageMatchInfos::operator=(const ImageMatchInfos &infos)
{
	if (this == &infos) {
		return *this;
	}

	avgDistance = infos.avgDistance;
	minDistance = infos.minDistance;
	confidence = infos.confidence;

	homography = infos.homography.clone();

	matches.resize(infos.matches.size());
	copy(infos.matches.begin(), infos.matches.end(), matches.begin());

	return *this;
}

ImageMatchInfos matchImages(const ImageDescriptor &sceneDescriptor, const ImageDescriptor &objectDescriptor)
{
	const float confidence = 0.4;
	ImageMatchInfos matchInfos;
	Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create("FlannBased");
	vector<vector<DMatch>> matches;

	matchInfos.minDistance = numeric_limits<float>::max();
	matchInfos.avgDistance = 0;

	descriptorMatcher->knnMatch(objectDescriptor.featureDescriptor, sceneDescriptor.featureDescriptor, matches, 2);

	for (int i = 0; i < matches.size(); ++i) {
		if (matches[i].size() < 2) {
			continue;
		}

		const DMatch &m0 = matches[i][0];
		const DMatch &m1 = matches[i][1];

		if (m0.distance < (1 - confidence) * m1.distance) {
			matchInfos.matches.push_back(make_pair(m0.trainIdx, m0.queryIdx));
			matchInfos.avgDistance += m0.distance;

			if (m0.distance < matchInfos.minDistance) {
				matchInfos.minDistance = m0.distance;
			}
		}
	}

	descriptorMatcher->knnMatch(sceneDescriptor.featureDescriptor, objectDescriptor.featureDescriptor, matches, 2);

	for (int i = 0; i < matches.size(); ++i) {
		if (matches[i].size() < 2) {
			continue;
		}

		const DMatch &m0 = matches[i][0];
		const DMatch &m1 = matches[i][1];

		if (m0.distance < (1 - confidence) * m1.distance) {
			if (find(matchInfos.matches.begin(), matchInfos.matches.end(), make_pair(m0.queryIdx, m0.trainIdx)) != matchInfos.matches.end()) {
				continue;
			}

			matchInfos.matches.push_back(make_pair(m0.queryIdx, m0.trainIdx));
			matchInfos.avgDistance += m0.distance;

			if (m0.distance < matchInfos.minDistance) {
				matchInfos.minDistance = m0.distance;
			}
		}
	}

	matchInfos.avgDistance /= matchInfos.matches.size();

	return matchInfos;
}

Mat computeHomography(const ImageDescriptor &sceneDescriptor, const ImageDescriptor &objectDescriptor, ImageMatchInfos &match)
{
	vector<Point2f> points[2];
	vector<pair<int, int>>::const_iterator it = match.matches.begin();

	while (it != match.matches.end()) {
		const pair<int, int> &m = *it++;

		Point2f scenePoint = sceneDescriptor.keypoints[m.first].pt;
		Point2f objectPoint = objectDescriptor.keypoints[m.second].pt;

		scenePoint.x -= sceneDescriptor.width / 2;
		scenePoint.y -= sceneDescriptor.height / 2;

		objectPoint.x -= objectDescriptor.width / 2;
		objectPoint.y -= objectDescriptor.height / 2;

		points[0].push_back(scenePoint);
		points[1].push_back(objectPoint);
	}

	vector<uchar> inliersMask;
	int numInliers = 0;
	Mat homography = findHomography(points[1], points[0], CV_RANSAC, 3.0, inliersMask);

	for (uchar mask : inliersMask) {
		if (mask) {
			++numInliers;
		}
	}

	float confidence = numInliers / (8.0 + 0.3 * match.matches.size());
	vector<uchar>::const_iterator inliersIt = inliersMask.cbegin();
	vector<Point2f>::const_iterator pointsIt[2];

	inliersIt = inliersMask.cbegin();
	pointsIt[0] = points[0].begin();
	pointsIt[1] = points[1].begin();

	while (inliersIt != inliersMask.cend()) {
		if (*inliersIt++) {
			pointsIt[0]++;
			pointsIt[1]++;
		} else {
			pointsIt[0] = points[0].erase(pointsIt[0]);
			pointsIt[1] = points[1].erase(pointsIt[1]);
		}
	}

	homography = findHomography(points[1], points[0], CV_RANSAC, 3.0);

	match.homography = homography;
	match.confidence = confidence;

	return homography;
}
