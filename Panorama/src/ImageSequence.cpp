#include "ImageSequence.h"

#include <set>

#include <opencv2/calib3d/calib3d.hpp>

#include "MatchGraph.h"

using namespace std;
using namespace cv;

namespace {
	bool matchImages(const ImageDescriptor &sceneDescriptor, const ImageDescriptor &objectDescriptor, ImageMatchInfos &matchInfos)
	{
		const double confidence = 0.6;
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

		homography = findHomography(points[1], points[0], CV_RANSAC, 3.0, inliersMask);
		matchesIt = match.matches.cbegin();
		inliersMaskIt = inliersMask.cbegin();

		int nbOverlaps = 0;
		int nbInliers = 0;

		while (matchesIt != match.matches.cend()) {
			Point2d point = objectDescriptor.keypoints[(*matchesIt++).second].pt;
			Mat pointH = Mat::ones(Size(1, 3), CV_64F);

			pointH.at<double>(0, 0) = point.x;
			pointH.at<double>(1, 0) = point.y;

			pointH = homography * pointH;
			point.x = pointH.at<double>(0, 0) / pointH.at<double>(2, 0) + sceneDescriptor.width / 2;
			point.y = pointH.at<double>(1, 0) / pointH.at<double>(2, 0) + sceneDescriptor.height / 2;

			if (point.x >= 0 && point.y >= 0 && point.x < sceneDescriptor.width && point.y < sceneDescriptor.height) {
				nbOverlaps++;

				if (*inliersMaskIt++) {
					nbInliers++;
				}
			}
		}

		match.nbInliers = nbInliers;
		match.nbOverlaps = nbOverlaps;
		match.homography = homography;
		match.confidence = match.nbInliers / (8.0 + 0.3 * match.nbOverlaps);
	}
}
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
void ImageSequence::addImage(int imageId, const ImagesRegistry &images)
{
	if (_keyFrames.empty()) {
		_keyFrames.push_back(imageId);
		return;
	}

	int lastKeyFrame = _keyFrames.back();
	ImageMatchInfos matchInfos;

	if (!matchImages(images.getDescriptor(lastKeyFrame), images.getDescriptor(imageId), matchInfos)) {
		_keyFrames.push_back(imageId);
		return;
	}

	computeHomography(images.getDescriptor(lastKeyFrame), images.getDescriptor(imageId), matchInfos);

	Mat_<double> translation = Mat_<double>::eye(3, 3);

	translation(0, 2) = -images.getDescriptor(imageId).width / 2;
	translation(1, 2) = -images.getDescriptor(imageId).height / 2;

	Mat_<double> H = translation.inv() * matchInfos.homography * translation;

	stringstream sstr;
	stringstream sstr2;

	sstr << imageId << " frame";
	sstr2 << imageId << " keyframe";

	namedWindow(sstr.str(), WINDOW_AUTOSIZE);
	namedWindow(sstr2.str(), WINDOW_AUTOSIZE);

	Mat img2;

	warpPerspective(images.getImage(imageId), img2, H, images.getImage(lastKeyFrame).size());

	imshow(sstr.str(), img2);
	imshow(sstr2.str(), images.getImage(lastKeyFrame));
}
