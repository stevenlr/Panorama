#include "ImageSequence.h"

#include <set>
#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "MatchGraph.h"
#include "Calibration.h"

using namespace std;
using namespace cv;

namespace {
	bool matchImages(const ImageDescriptor &sceneDescriptor, const ImageDescriptor &objectDescriptor, ImageMatchInfos &matchInfos)
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
				matchInfos.matches.push_back(make_pair(m0.trainIdx, m0.queryIdx));
			}
		}

		if (matchInfos.matches.size() < 4) {
			return false;
		}

		return true;
	}

	void computeHomography(const ImageDescriptor &sceneDescriptor, const ImageDescriptor &objectDescriptor, ImageMatchInfos &match)
	{
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

		vector<uchar> inliersMask;
		Mat homography = findHomography(points[1], points[0], CV_RANSAC, 3.0, inliersMask);
		vector<uchar>::const_iterator inliersMaskIt;
		vector<Point2f>::const_iterator pointsIt[2];

		inliersMaskIt = inliersMask.cbegin();
		pointsIt[0] = points[0].begin();
		pointsIt[1] = points[1].begin();

		while (inliersMaskIt != inliersMask.cend()) {
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
		inliersMaskIt = inliersMask.cbegin();

		int nbOverlaps = 0;
		int nbInliers = 0;
		vector<uchar>::const_iterator inliersMask2It = inliersMask2.cbegin();

		while (matchesIt != match.matches.cend()) {
			if (*inliersMaskIt++) {
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
					}
				}

				inliersMask2It++;
			}

			matchesIt++;
		}

		match.nbInliers = nbInliers;
		match.nbOverlaps = nbOverlaps;
		match.homography = homography;
		match.confidence = match.nbInliers / (8.0 + 0.3 * match.nbOverlaps);
	}
}

void ImageSequence::addImage(int imageId, const ImagesRegistry &images)
{
	_nbFrames++;

	if (_keyFrames.empty()) {
		_keyFrames.push_back(imageId);
		_homographies.push_back(Mat::eye(Size(3, 3), CV_64F));
		return;
	}

	int lastKeyFrame = _keyFrames.back();
	ImageMatchInfos matchInfos;

	if (!matchImages(images.getDescriptor(lastKeyFrame), images.getDescriptor(imageId), matchInfos)) {
		_keyFrames.push_back(imageId);
		_homographies.push_back(Mat::eye(Size(3, 3), CV_64F));
		return;
	}

	computeHomography(images.getDescriptor(lastKeyFrame), images.getDescriptor(imageId), matchInfos);

	Mat_<double> translation = Mat_<double>::eye(3, 3);

	translation(0, 2) = -images.getDescriptor(imageId).width / 2;
	translation(1, 2) = -images.getDescriptor(imageId).height / 2;

	Mat_<double> H = translation.inv() * matchInfos.homography * translation;
	Mat mask = Mat::ones(images.getImage(imageId).size(), CV_8U);
	const Mat &keyImage = images.getImage(lastKeyFrame);

	warpPerspective(mask, mask, H, keyImage.size());

	int nbOverlap = 0;

	for (int y = 0; y < mask.size().height; ++y) {
		const uchar *ptr = mask.ptr<uchar>(y);

		for (int x = 0; x < mask.size().width; ++x) {
			if (*ptr++) {
				nbOverlap++;
			}
		}
	}

	float overlapRatio = static_cast<float>(nbOverlap) / (mask.size().width * mask.size().height);

	mask = Mat::ones(images.getImage(lastKeyFrame).size(), CV_8U);
	const Mat &image = images.getImage(imageId);

	translation(0, 2) = -images.getDescriptor(lastKeyFrame).width / 2;
	translation(1, 2) = -images.getDescriptor(lastKeyFrame).height / 2;

	H = translation.inv() * matchInfos.homography.inv() * translation;
	warpPerspective(mask, mask, H, image.size());

	nbOverlap = 0;

	for (int y = 0; y < mask.size().height; ++y) {
		const uchar *ptr = mask.ptr<uchar>(y);

		for (int x = 0; x < mask.size().width; ++x) {
			if (*ptr++) {
				nbOverlap++;
			}
		}
	}

	float overlapRatio2 = static_cast<float>(nbOverlap) / (mask.size().width * mask.size().height);

	if (std::min(overlapRatio, overlapRatio2) < 0.6) {
		_keyFrames.push_back(imageId);
		_homographies.push_back(Mat::eye(Size(3, 3), CV_64F));
	} else {
		_homographies.push_back(matchInfos.homography);
	}

	findFocalLength(matchInfos.homography, _focalLengths);
}

int ImageSequence::getNbKeyframes() const
{
	return _keyFrames.size();
}

int ImageSequence::getKeyFrame(int i) const
{
	return _keyFrames[i];
}

double ImageSequence::estimateFocalLength()
{
	assert(_focalLengths.size() != 0);
	return getMedianFocalLength(_focalLengths);
}

void ImageSequence::addIntermediateFramesToScene(Scene &scene)
{
	int numKeyFrame = 0;

	for (int numFrame = 0; numFrame < _nbFrames; ++numFrame) {
		if (numKeyFrame < _keyFrames.size() - 1) {
			if (_keyFrames[numKeyFrame + 1] == numFrame) {
				numKeyFrame++;
				continue;
			}
		}

		if (_keyFrames[numKeyFrame] != numFrame) {
			scene.addImage(numFrame);
			scene.setParent(scene.getIdInScene(numFrame), _keyFrames[numKeyFrame]);
			scene.setTransform(scene.getIdInScene(numFrame), _homographies[numFrame]);
		}
	}
}
