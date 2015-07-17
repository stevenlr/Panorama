#include "ImageRegistration.h"

#include <set>

#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;

void computeHomography(const ImageDescriptor &sceneDescriptor, const ImageDescriptor &objectDescriptor, const ImageMatchInfos &match, ComputeHomographyOutput &output)
{
	vector<Point2f> points[2];
	vector<pair<int, int>>::const_iterator matchesIt = match.matches.cbegin();

	while (matchesIt != match.matches.cend()) {
		const pair<int, int> &m = *matchesIt++;

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

	Mat homography = findHomography(points[1], points[0], CV_RANSAC, 3.0, inliersMask);

	vector<uchar>::iterator inliersMaskIt;
	vector<Point2f>::const_iterator pointsIt[2];

	inliersMaskIt = inliersMask.begin();
	pointsIt[0] = points[0].begin();
	pointsIt[1] = points[1].begin();

	while (inliersMaskIt != inliersMask.end()) {
		if (*inliersMaskIt++) {
			pointsIt[0]++;
			pointsIt[1]++;
		} else {
			pointsIt[0] = points[0].erase(pointsIt[0]);
			pointsIt[1] = points[1].erase(pointsIt[1]);
		}
	}

	vector<uchar> inliersMask2;

	homography = findHomography(points[1], points[0], CV_RANSAC, 3.0, inliersMask2);

	matchesIt = match.matches.cbegin();
	inliersMaskIt = inliersMask.begin();

	int nbOverlaps = 0;
	int nbInliers = 0;
	vector<uchar>::const_iterator inliersMask2It = inliersMask2.cbegin();

	while (matchesIt != match.matches.cend()) {
		if (*inliersMaskIt) {
			Point2d point = objectDescriptor.keypoints[matchesIt->second].pt;
			Mat pointH = Mat::ones(Size(1, 3), CV_64F);

			pointH.at<double>(0, 0) = point.x - objectDescriptor.width / 2;
			pointH.at<double>(1, 0) = point.y - objectDescriptor.height / 2;

			pointH = homography * pointH;
			point.x = pointH.at<double>(0, 0) / pointH.at<double>(2, 0) + objectDescriptor.width / 2;
			point.y = pointH.at<double>(1, 0) / pointH.at<double>(2, 0) + objectDescriptor.height / 2;

			if (point.x >= 0 && point.y >= 0 && point.x < sceneDescriptor.width && point.y < sceneDescriptor.height) {
				nbOverlaps++;

				if (*inliersMask2It) {
					nbInliers++;
				} else {
					*inliersMaskIt = 0;
				}
			}

			inliersMask2It++;
		}

		matchesIt++;
		inliersMaskIt++;
	}

	output.nbInliers = nbInliers;
	output.nbOverlaps = nbOverlaps;
	output.inliersMask = inliersMask;
	output.homography = homography;
}

bool matchImages(const ImageDescriptor &sceneDescriptor, const ImageDescriptor &objectDescriptor, ImageMatchInfos &matchInfos, std::mutex *matchMutex)
{
	const double confidence = 0.5;
	Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create("BruteForce");
	vector<vector<DMatch>> matches;
	set<pair<int, int>> matchesSet;

	descriptorMatcher->knnMatch(objectDescriptor.featureDescriptor, sceneDescriptor.featureDescriptor, matches, 2);
	descriptorMatcher->clear();

	for (size_t i = 0; i < matches.size(); ++i) {
		if (matches[i].size() < 2) {
			continue;
		}

		const DMatch &m0 = matches[i][0];
		const DMatch &m1 = matches[i][1];

		if (m0.distance < (1 - confidence) * m1.distance) {
			matchesSet.insert(make_pair(m0.trainIdx, m0.queryIdx));

			if (matchMutex) {
				matchMutex->lock();
			}

			matchInfos.matches.push_back(make_pair(m0.trainIdx, m0.queryIdx));

			if (matchMutex) {
				matchMutex->unlock();
			}
		}
	}

	if (matchMutex) {
		matchMutex->lock();
	}

	if (matchInfos.matches.size() < 4) {
		if (matchMutex) {
			matchMutex->unlock();
		}
		return false;
	}

	if (matchMutex) {
		matchMutex->unlock();
	}

	return true;
}
