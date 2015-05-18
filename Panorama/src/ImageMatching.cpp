#include "ImageMatching.h"

#include "Calibration.h"

using namespace std;
using namespace cv;

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
	const double confidence = 0.4;
	ImageMatchInfos matchInfos;
	Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create("FlannBased");
	vector<vector<DMatch>> matches;

	matchInfos.minDistance = numeric_limits<float>::max();
	matchInfos.avgDistance = 0;

	descriptorMatcher->knnMatch(objectDescriptor.featureDescriptor, sceneDescriptor.featureDescriptor, matches, 2);

	for (size_t i = 0; i < matches.size(); ++i) {
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
	int numInliers = 0, numOverlap = 0;
	Mat homography = findHomography(points[1], points[0], CV_RANSAC, 3.0, inliersMask);

	for (uchar mask : inliersMask) {
		if (mask) {
			++numInliers;
		}
	}

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
			++numOverlap;
		}
	}

	match.homography = homography;
	match.confidence = numInliers / (8.0 + 0.3 * numOverlap);

	return homography;
}

void extractFeatures(const Scene &scene, vector<ImageDescriptor> &descriptors)
{
	Ptr<FeatureDetector> featureDetector = FeatureDetector::create("SIFT");
	Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create("SIFT");

	for (int i = 0; i < scene.getNbImages(); ++i) {
		descriptors[i].image = i;
		descriptors[i].width = scene.getImage(i).size().width;
		descriptors[i].height = scene.getImage(i).size().height;
		featureDetector->detect(scene.getImageBW(i), descriptors[i].keypoints);
		descriptorExtractor->compute(scene.getImageBW(i), descriptors[i].keypoints, descriptors[i].featureDescriptor);
	}
}

void pairwiseMatch(const Scene &scene,
				   const vector<ImageDescriptor> &descriptors,
				   list<MatchGraphEdge> &matchGraphEdges,
				   map<pair<int, int>, ImageMatchInfos> &matchInfosMap,
				   vector<double> &focalLengths)
{
	int nbImages = scene.getNbImages();

	for (int i = 0; i < nbImages; ++i) {
		for (int j = i + 1; j < nbImages; ++j) {
			if (i == j) {
				continue;
			}

			ImageMatchInfos matchInfos = matchImages(descriptors[j], descriptors[i]);

			if (matchInfos.matches.size() >= 4) {
				computeHomography(descriptors[j], descriptors[i], matchInfos);

				if (matchInfos.confidence > 1) {
					ImageMatchInfos matchInfos2 = matchInfos;

					matchInfos2.homography = matchInfos2.homography.inv();

					findFocalLength(matchInfos.homography, focalLengths);
					findFocalLength(matchInfos2.homography, focalLengths);

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

					matchInfosMap[make_pair(i, j)] = matchInfos;
					matchInfosMap[make_pair(j, i)] = matchInfos2;
				}
			}
		}
	}
}
