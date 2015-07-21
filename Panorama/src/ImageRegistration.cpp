#include "ImageRegistration.h"

#include <iostream>
#include <set>

#include <opencv2/calib3d/calib3d.hpp>

#include "Configuration.h"

using namespace std;
using namespace cv;

namespace {
	Mat homographyToParameters(Mat homography)
	{
		Mat_<double> parameters(8, 1);

		parameters(0, 0) = homography.at<double>(0, 0);
		parameters(1, 0) = homography.at<double>(1, 0);
		parameters(2, 0) = homography.at<double>(2, 0);
		parameters(3, 0) = homography.at<double>(0, 1);
		parameters(4, 0) = homography.at<double>(1, 1);
		parameters(5, 0) = homography.at<double>(2, 1);
		parameters(6, 0) = homography.at<double>(0, 2);
		parameters(7, 0) = homography.at<double>(1, 2);

		return parameters;
	}

	Mat parametersToHomography(Mat parameters)
	{
		Mat_<double> homography(3, 3);

		homography(0, 0) = parameters.at<double>(0, 0);
		homography(1, 0) = parameters.at<double>(1, 0);
		homography(2, 0) = parameters.at<double>(2, 0);
		homography(0, 1) = parameters.at<double>(3, 0);
		homography(1, 1) = parameters.at<double>(4, 0);
		homography(2, 1) = parameters.at<double>(5, 0);
		homography(0, 2) = parameters.at<double>(6, 0);
		homography(1, 2) = parameters.at<double>(7, 0);
		homography(2, 2) = 1;

		return homography;
	}

	Mat computeResidues(vector<pair<Point2d, Point2d>> &matchPoints, Mat homography)
	{
		Mat residues = Mat::zeros(Size(1, matchPoints.size() * 2), CV_64F);
		Mat_<double> pointH = Mat::ones(Size(1, 3), CV_64F);
		int i = 0;

		for (auto &m : matchPoints)
		{
			pointH(0, 0) = m.first.x;
			pointH(1, 0) = m.first.y;
			pointH(2, 0) = 1;

			pointH = homography * pointH;
			pointH /= pointH(2, 0);

			Point2d transformedPoint(pointH(0, 0), pointH(1, 0));
			Point2d diff = m.second - transformedPoint;

			residues.at<double>(i++, 0) = diff.x;
			residues.at<double>(i++, 0) = diff.y;
		}

		return residues;
	}

	double computeError(vector<pair<Point2d, Point2d>> &matchPoints, Mat homography)
	{
		double dist = 0;
		Mat_<double> pointH = Mat::ones(Size(1, 3), CV_64F);

		for (auto &m : matchPoints)
		{
			pointH(0, 0) = m.first.x;
			pointH(1, 0) = m.first.y;
			pointH(2, 0) = 1;

			pointH = homography * pointH;
			pointH /= pointH(2, 0);

			Point2d transformedPoint(pointH(0, 0), pointH(1, 0));
			Point2d diff = m.second - transformedPoint;

			dist += diff.x * diff.x + diff.y * diff.y;
		}

		return dist;
	}

	Mat computeJacobian(vector<pair<Point2d, Point2d>> &matchPoints, Mat parameters)
	{
		Mat J(Size(8, matchPoints.size() * 2), CV_64F);
		const double eps = 0.01;

		for (int i = 0; i < 8; ++i) {
			Mat res;

			parameters.at<double>(i, 0) += eps;
			res = computeResidues(matchPoints, parametersToHomography(parameters));

			parameters.at<double>(i, 0) -= 2 * eps;
			res = res - computeResidues(matchPoints, parametersToHomography(parameters));
			res /= 2 * eps;

			res.copyTo(J.col(i));

			parameters.at<double>(i, 0) += eps;
		}

		return J;
	}

	Mat optimizeHomography(vector<pair<Point2d, Point2d>> &matchPoints, Mat homography)
	{
		Mat parameters = homographyToParameters(homography);
		double lambda = 1;
		double v = 10;
		float firstError = computeError(matchPoints, homography);
		float lastError = firstError;
		float error = lastError;
		int nbIterations = Configuration::getInstance()->getRegistrationOptimizationIterations();
		
		for (int i = 0; i < nbIterations; ++i) {
			Mat residues = computeResidues(matchPoints, parametersToHomography(parameters));
			Mat J = computeJacobian(matchPoints, parameters);
			Mat JtJ = J.t() * J;
			Mat Jtr = J.t() * residues;
			Mat invTerm = JtJ + lambda * Mat::diag(JtJ.diag());

			invTerm = -invTerm.inv();

			Mat incr = invTerm * Jtr;

			error = computeError(matchPoints, parametersToHomography(parameters + incr));

			if (lastError < error) {
				lambda *= v;
			} else {
				lambda /= v;
				parameters += incr;
				lastError = error;
			}
		}

		if (firstError - error > 0) {
			return parametersToHomography(parameters);
		} else {
			return homography;
		}
	}
}

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
	vector<pair<Point2d, Point2d>> matchPoints;
	Point2f objectOffset(objectDescriptor.width / 2, objectDescriptor.height / 2);
	Point2f sceneOffset(sceneDescriptor.width / 2, sceneDescriptor.height / 2);

	while (matchesIt != match.matches.cend()) {
		if (*inliersMaskIt) {
			Point2f point = objectDescriptor.keypoints[matchesIt->second].pt;
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
					matchPoints.push_back(make_pair(objectDescriptor.keypoints[matchesIt->second].pt - objectOffset,
						sceneDescriptor.keypoints[matchesIt->first].pt - sceneOffset));
				} else {
					*inliersMaskIt = 0;
				}
			}

			inliersMask2It++;
		}

		matchesIt++;
		inliersMaskIt++;
	}

	//homography = optimizeHomography(matchPoints, homography);

	output.nbInliers = nbInliers;
	output.nbOverlaps = nbOverlaps;
	output.inliersMask = inliersMask;
	output.homography = homography;
}

bool matchImages(const ImageDescriptor &sceneDescriptor, const ImageDescriptor &objectDescriptor, ImageMatchInfos &matchInfos, std::mutex *matchMutex)
{
	const double confidence = Configuration::getInstance()->getFeatureMatchConfidence();
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
