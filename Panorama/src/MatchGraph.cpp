#include "MatchGraph.h"

#include <list>
#include <algorithm>

#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;

ImageMatchInfos::ImageMatchInfos()
{
	confidence = 0;
	nbInliers = 0;
	nbOverlaps = 0;

	homography = Mat();
}

ImageMatchInfos::ImageMatchInfos(const ImageMatchInfos &infos)
{
	confidence = infos.confidence;
	nbInliers = infos.nbInliers;
	nbOverlaps = infos.nbOverlaps;

	homography = infos.homography.clone();

	matches.resize(infos.matches.size());
	copy(infos.matches.begin(), infos.matches.end(), matches.begin());

	inliersMask.resize(infos.inliersMask.size());
	copy(infos.inliersMask.begin(), infos.inliersMask.end(), inliersMask.begin());
}

ImageMatchInfos &ImageMatchInfos::operator=(const ImageMatchInfos &infos)
{
	if (this == &infos) {
		return *this;
	}

	confidence = infos.confidence;
	nbInliers = infos.nbInliers;
	nbOverlaps = infos.nbOverlaps;

	homography = infos.homography.clone();

	matches.resize(infos.matches.size());
	copy(infos.matches.begin(), infos.matches.end(), matches.begin());

	inliersMask.resize(infos.inliersMask.size());
	copy(infos.inliersMask.begin(), infos.inliersMask.end(), inliersMask.begin());

	return *this;
}

MatchGraph::MatchGraph(const ImagesRegistry &images)
{
	int nbImages = images.getNbImages();
	list<MatchGraphEdge> matchGraphEdges;

	_matchInfos.resize(nbImages);

	for (int i = 0; i < nbImages; ++i) {
		_matchInfos[i].resize(nbImages);
	}

	for (int i = 0; i < nbImages; ++i) {
		for (int j = i + 1; j < nbImages; ++j) {
			if (i == j) {
				continue;
			}
			
			const ImageDescriptor &sceneDescriptor = images.getDescriptor(i);
			const ImageDescriptor &objectDescriptor = images.getDescriptor(j);

			if (matchImages(sceneDescriptor, objectDescriptor)) {
				computeHomography(sceneDescriptor, objectDescriptor);

				const ImageMatchInfos &matchInfos = _matchInfos[i][j];

				if (matchInfos.confidence > 1) {
					ImageMatchInfos &matchInfos2 = _matchInfos[j][i];

					matchInfos2.homography = matchInfos2.homography.inv();

					MatchGraphEdge edge1;
					MatchGraphEdge edge2;

					edge1.objectImage = i;
					edge1.sceneImage = j;
					edge1.confidence = matchInfos.confidence;
					edge2.objectImage = i;
					edge2.sceneImage = j;
					edge2.confidence = matchInfos.confidence;

					matchGraphEdges.push_back(edge1);
					matchGraphEdges.push_back(edge2);
				}
			}
		}
	}
}

bool MatchGraph::matchImages(const ImageDescriptor &sceneDescriptor, const ImageDescriptor &objectDescriptor)
{
	const double confidence = 0.4;
	ImageMatchInfos &matchInfos = _matchInfos[sceneDescriptor.image][objectDescriptor.image];
	Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create("FlannBased");
	vector<vector<DMatch>> matches;

	descriptorMatcher->knnMatch(objectDescriptor.featureDescriptor, sceneDescriptor.featureDescriptor, matches, 2);

	for (size_t i = 0; i < matches.size(); ++i) {
		if (matches[i].size() < 2) {
			continue;
		}

		const DMatch &m0 = matches[i][0];
		const DMatch &m1 = matches[i][1];

		if (m0.distance < (1 - confidence) * m1.distance) {
			matchInfos.matches.push_back(make_pair(m0.trainIdx, m0.queryIdx));
		}
	}

	descriptorMatcher->knnMatch(sceneDescriptor.featureDescriptor, objectDescriptor.featureDescriptor, matches, 2);

	if (matches.size() < 4) {
		return false;
	}

	for (size_t i = 0; i < matches.size(); ++i) {
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
		}
	}

	return true;
}

void MatchGraph::computeHomography(const ImageDescriptor &sceneDescriptor, const ImageDescriptor &objectDescriptor)
{
	ImageMatchInfos &match = _matchInfos[sceneDescriptor.image][objectDescriptor.image];
	vector<Point2f> points[2];
	vector<pair<int, int>>::const_iterator matchesIt = match.matches.cbegin();

	match.nbInliers = 0;
	match.nbOverlaps = 0;
	match.inliersMask.clear();

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

	Mat homography = findHomography(points[1], points[0], CV_RANSAC, 3.0, match.inliersMask);

	for (uchar mask : match.inliersMask) {
		if (mask) {
			match.nbInliers++;
		}
	}

	vector<uchar>::const_iterator inliersIt;
	vector<Point2f>::const_iterator pointsIt[2];

	inliersIt = match.inliersMask.cbegin();
	pointsIt[0] = points[0].begin();
	pointsIt[1] = points[1].begin();

	while (inliersIt != match.inliersMask.cend()) {
		if (*inliersIt++) {
			pointsIt[0]++;
			pointsIt[1]++;
		} else {
			pointsIt[0] = points[0].erase(pointsIt[0]);
			pointsIt[1] = points[1].erase(pointsIt[1]);
		}
	}

	homography = findHomography(points[1], points[0], CV_RANSAC, 3.0);

	matchesIt = match.matches.cbegin();

	while (matchesIt != match.matches.cend()) {
		Point2d point = objectDescriptor.keypoints[(*matchesIt++).second].pt;
		Mat pointH = Mat::ones(Size(1, 3), CV_64F);

		pointH.at<double>(0, 0) = point.x;
		pointH.at<double>(1, 0) = point.y;

		pointH = homography * pointH;
		point.x = pointH.at<double>(0, 0) / pointH.at<double>(2, 0);
		point.y = pointH.at<double>(1, 0) / pointH.at<double>(2, 0);

		if (point.x >= 0 && point.y >= 0 && point.x < sceneDescriptor.width && point.y < sceneDescriptor.height) {
			match.nbOverlaps++;
		}
	}

	match.homography = homography;
	match.confidence = match.nbInliers / (8.0 + 0.3 * match.nbOverlaps);
}
