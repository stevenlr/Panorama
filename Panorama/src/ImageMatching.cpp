#include "ImageMatching.h"

using namespace std;
using namespace cv;

bool compareMatchMatrixElements(const MatchMatrixElement &first, const MatchMatrixElement &second)
{
	return first.second.avgDistance < second.second.avgDistance;
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

Mat computeHomography(const ImageDescriptor &sceneDescriptor, const ImageDescriptor &objectDescriptor, const ImageMatchInfos &match)
{
	vector<Point2f> points[2];
	vector<pair<int, int>>::const_iterator it = match.matches.begin();

	while (it != match.matches.end()) {
		const pair<int, int> &m = *it++;

		points[0].push_back(sceneDescriptor.keypoints[m.first].pt);
		points[1].push_back(objectDescriptor.keypoints[m.second].pt);
	}

	return findHomography(points[1], points[0], CV_RANSAC);
}
