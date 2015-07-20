#include "ImageSequence.h"

#include <set>
#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "MatchGraph.h"
#include "Calibration.h"
#include "RegistrationOptimization.h"
#include "ImageRegistration.h"

using namespace std;
using namespace cv;

namespace {
	bool matchImages(const ImageDescriptor &sceneDescriptor, const ImageDescriptor &objectDescriptor, ImageMatchInfos &matchInfos)
	{
		return ::matchImages(sceneDescriptor, objectDescriptor, matchInfos, nullptr);
	}

	void computeHomography(const ImageDescriptor &sceneDescriptor, const ImageDescriptor &objectDescriptor, ImageMatchInfos &match)
	{
		ComputeHomographyOutput output;

		::computeHomography(sceneDescriptor, objectDescriptor, match, output);

		match.inliersMask = output.inliersMask;
		match.nbInliers = output.nbInliers;
		match.nbOverlaps = output.nbOverlaps;
		match.homography = output.homography;
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

	if (std::min(overlapRatio, overlapRatio2) < 0.75) {
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
