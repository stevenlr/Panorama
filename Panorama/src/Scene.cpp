#include "Scene.h"

#include <set>
#include <queue>
#include <algorithm>
#include <iostream>
#include <ctime>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/stitching/detail/warpers.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "Constants.h"
#include "Calibration.h"

using namespace std;
using namespace cv;

Scene::Scene()
{
	_estimatedFocalLength = -1;
	_nbImages = 0;
}

void Scene::setEstimatedFocalLength(double f)
{
	_estimatedFocalLength = f;
}

void Scene::addImage(int id)
{
	_reverseIds.insert(make_pair(id, _nbImages++));
	_images.push_back(id);
	_transform.push_back(Mat());
	_parent.push_back(-1);
}

int Scene::getNbImages() const
{
	return _nbImages;
}

int Scene::getImage(int id) const
{
	assert(id >= 0 && id < _nbImages);
	return _images[id];
}

int Scene::getIdInScene(int image) const
{
	map<int, int>::const_iterator it = _reverseIds.find(image);

	if (it == _reverseIds.cend()) {
		return -1;
	}

	return it->second;
}

int Scene::getParent(int image) const
{
	return _parent[image];
}

void Scene::setParent(int image, int parent)
{
	assert(image >= 0 && image < _nbImages);
	assert(parent >= -1 && parent < _nbImages);
	assert(image != parent);

	_parent[image] = parent;
}

void Scene::setTransform(int image, const Mat &transform)
{
	assert(image >= 0 && image < _nbImages);
	assert(transform.size() == Size(3, 3));

	_transform[image] = transform;
}

const cv::Mat &Scene::getTransform(int image) const
{
	assert(image >= 0 && image < _nbImages);

	return _transform[image];
}

Mat Scene::getFullTransform(int image) const
{
	assert(image >= 0 && image < _nbImages);

	if (_parent[image] != -1) {
		return getFullTransform(_parent[image]) * _transform[image];
	} else {
		return _transform[image].clone();
	}
}

namespace {
	pair<Point2i, Point2i> getOverlappingRegion(const pair<Point2i, Point2i> &a, const pair<Point2i, Point2i> &b)
	{
		Point2i maxCorner, minCorner;

		minCorner.x = std::max(a.first.x, b.first.x);
		minCorner.y = std::max(a.first.y, b.first.y);

		maxCorner.x = std::min(a.second.x, b.second.x);
		maxCorner.y = std::min(a.second.y, b.second.y);

		return make_pair(minCorner, maxCorner);
	}

	float getWeight(int x, int size) {
		return 1 - std::abs((static_cast<float>(x) / size) * 2 - 1);
	}
}

#define WEIGHT_MAX 1000.0f

Mat Scene::composePanoramaSpherical(const ImagesRegistry &images, int scale)
{
	vector<Mat> warpedImages(_nbImages);
	vector<Mat> warpedMasks(_nbImages);
	vector<Mat> warpedWeights(_nbImages);
	vector<Point2d> corners(_nbImages);
	clock_t start;
	float elapsedTime;

	if (_nbImages < 2) {
		return Mat();
	}

	cout << "  Warping images";

	detail::SphericalWarper warper(static_cast<float>(scale));

	start = clock();
	for (int i = 0; i < _nbImages; ++i) {
		Mat img = images.getImage(getImage(i));
		Size size = img.size();
		
		Mat homography = getFullTransform(i).clone();

		if (homography.size() != Size(3, 3)) {
			continue;
		}

		cout << ".";

		Mat cameraParameters = Mat::eye(Size(3, 3), CV_64F);

		cameraParameters.at<double>(0, 0) = _estimatedFocalLength;
		cameraParameters.at<double>(1, 1) = _estimatedFocalLength;

		homography = cameraParameters.inv() * homography * cameraParameters;

		cameraParameters.at<double>(0, 2) = size.width / 2;
		cameraParameters.at<double>(1, 2) = size.height / 2;

		Mat R, Rvect;

		Rodrigues(homography, Rvect);
		Rodrigues(Rvect, R);

		R.convertTo(R, CV_32F);
		cameraParameters.convertTo(cameraParameters, CV_32F);

		corners[i] = warper.warp(img, cameraParameters, R, INTER_LINEAR, BORDER_CONSTANT, warpedImages[i]);

		Mat maskNormal = Mat::ones(size, CV_8U);

		warper.warp(maskNormal, cameraParameters, R, INTER_NEAREST, BORDER_CONSTANT, warpedMasks[i]);
		distanceTransform(warpedMasks[i], warpedWeights[i], CV_DIST_L1, 3);
	}

	elapsedTime = static_cast<float>(clock() - start) / _nbImages / CLOCKS_PER_SEC;
	cout << endl << "  Warping average: " << elapsedTime << "s" << endl;

	/*Mat overlapIntensities(Size(_nbImages, _nbImages), CV_64F, Scalar(0));
	Mat overlapSizes(Size(_nbImages, _nbImages), CV_32S, Scalar(0));

	cout << "  Compensating exposure" << endl;
	start = clock();

	for (int i = 0; i < _nbImages; ++i) {
		for (int j = i; j < _nbImages; ++j) {
			pair<Point2i, Point2i> overlap = getOverlappingRegion(corners[i], corners[j]);

			if (overlap.first.x >= overlap.second.x || overlap.first.y >= overlap.second.y) {
				continue;
			}

			Rect region(overlap.first.x, overlap.first.y, overlap.second.x - overlap.first.x, overlap.second.y - overlap.first.y);
			Mat mask0(warpedMasks[i], region);
			Mat mask1(warpedMasks[j], region);
			Mat image0(warpedImages[i], region);
			Mat image1(warpedImages[j], region);
			int overlapSize = 0;
			double overlapIntensity0 = 0;
			double overlapIntensity1 = 0;

			for (int y = 0; y < region.height; ++y) {
				uchar *mask0ptr = mask0.ptr<uchar>(y);
				uchar *mask1ptr = mask1.ptr<uchar>(y);
				uchar *image0ptr = image0.ptr<uchar>(y);
				uchar *image1ptr = image1.ptr<uchar>(y);

				for (int x = 0; x < region.width; ++x) {
					if (*mask0ptr++ != 0 && *mask1ptr++ != 0) {
						++overlapSize;
						overlapIntensity0 += saturate_cast<uchar>((*image0ptr++ + *image0ptr++ + *image0ptr++) / 3);
						overlapIntensity1 += saturate_cast<uchar>((*image1ptr++ + *image1ptr++ + *image1ptr++) / 3);
					}
				}
			}

			overlapSizes.at<int>(i, j) = overlapSize;
			overlapSizes.at<int>(j, i) = overlapSize;
			overlapIntensities.at<double>(i, j) = overlapIntensity0 / overlapSize;
			overlapIntensities.at<double>(j, i) = overlapIntensity1 / overlapSize;
		}
	}

	vector<double> gains(_nbImages);

	{
		Mat A(Size(_nbImages, _nbImages), CV_64F);
		Mat b(Size(1, _nbImages), CV_64F);
		double gainDeviationFactor = 1.0 / (0.1 * 0.1);
		double errorDeviationFactor = 1.0 / (10 * 10);

		A.setTo(0);
		b.setTo(0);

		for (int i = 0; i < _nbImages; ++i) {
			for (int j = 0; j < _nbImages; ++j) {
				int N = overlapSizes.at<int>(i, j);

				b.at<double>(i, 0) += gainDeviationFactor * N;
				A.at<double>(i, i) += gainDeviationFactor * N;

				if (i != j) {
					double Iij = overlapIntensities.at<double>(i, j);
					double Iji = overlapIntensities.at<double>(j, i);

					A.at<double>(i, i) += 2 * Iij * Iij * errorDeviationFactor * N;
					A.at<double>(i, j) -= 2 * Iij * Iji * errorDeviationFactor * N;
				}
			}
		}

		Mat x;
		solve(A, b, x);

		for (int i = 0; i < _nbImages; ++i) {
			gains[i] = x.at<double>(i, 0);
			warpedImages[i] *= gains[i];
		}
	}

	elapsedTime = static_cast<float>(clock() - start) / _nbImages / CLOCKS_PER_SEC;
	cout << "  Gain compensation average: " << elapsedTime << "s" << endl;*/

	cout << "  Building weight masks" << endl;

	start = clock();

	for (int i = 0; i < _nbImages; ++i) {
		Size sizeI = warpedWeights[i].size();

		for (int j = i + 1; j < _nbImages; ++j) {
			Size sizeJ = warpedWeights[j].size();

			if (corners[i].x + sizeI.width <= corners[j].x
				|| corners[j].x + sizeJ.width <= corners[i].x
				|| corners[i].y + sizeI.height <= corners[j].y
				|| corners[j].y + sizeJ.height <= corners[i].y) {
				continue;
			}

			Point2d corner1;

			corner1.x = std::max(corners[i].x, corners[j].x);
			corner1.y = std::max(corners[i].y, corners[j].y);

			Point2d corner2;

			corner2.x = std::min(corners[i].x + sizeI.width, corners[j].x + sizeJ.width);
			corner2.y = std::min(corners[i].y + sizeI.height, corners[j].y + sizeJ.height);

			Size size = corner2 - corner1;

			Mat roiI(warpedWeights[i], Rect(corner1 - corners[i], size));
			Mat roiJ(warpedWeights[j], Rect(corner1 - corners[j], size));

			for (int y = 0; y < size.height; ++y) {
				float *ptrI = roiI.ptr<float>(y);
				float *ptrJ = roiJ.ptr<float>(y);

				for (int x = 0; x < size.width; ++x) {
					if (*ptrI > *ptrJ) {
						*ptrJ = 0;
					} else {
						*ptrI = 0;
					}

					++ptrI;
					++ptrJ;
				}
			}
		}
	}

	elapsedTime = static_cast<float>(clock() - start) / _nbImages / CLOCKS_PER_SEC;
	cout << "  Weight mask building average: " << elapsedTime << "s" << endl;

	cout << "  Multiband blending" << endl;

	start = clock();

	Ptr<detail::Blender> blender = detail::Blender::createDefault(detail::Blender::MULTI_BAND);

	(reinterpret_cast<detail::MultiBandBlender *>(&blender))->setNumBands(5);

	{
		vector<Point> blendCorners(_nbImages);
		vector<Size> blendSizes(_nbImages);

		for (int i = 0; i < _nbImages; ++i) {
			blendCorners[i] = corners[i];
			blendSizes[i] = warpedImages[i].size();
		}

		blender->prepare(blendCorners, blendSizes);
	}

	for (int i = 0; i < _nbImages; ++i) {
		Mat mask;

		warpedWeights[i].convertTo(mask, CV_8U);
		blender->feed(warpedImages[i], mask * 255, corners[i]);
	}

	Mat finalImage;

	blender->blend(finalImage, Mat());
	finalImage.convertTo(finalImage, CV_8UC3);

	elapsedTime = static_cast<float>(clock() - start) / CLOCKS_PER_SEC;
	cout << "  Multiband blending total: " << elapsedTime << "s" << endl;

	return finalImage;
}

/*Mat Scene::composePanoramaPlanar()
{
	vector<Point2f> srcPoints(4);
	vector<Point2f> dstPoints(4);
	vector<Mat> fullTransforms(_nbImages);
	float maxX = numeric_limits<float>::min();
	float minX = numeric_limits<float>::max();
	float maxY = numeric_limits<float>::min();
	float minY = numeric_limits<float>::max();

	srcPoints[0] = Point2f(0, 0);

	for (int i = 0; i < _nbImages; ++i) {
		Mat fullTransform = getFullTransform(i);
		const Size &size = getImage(i).size();
		Mat translation = Mat::eye(Size(3, 3), CV_64F);

		translation.at<double>(0, 2) = -size.width / 2;
		translation.at<double>(1, 2) = -size.height / 2;

		fullTransform = fullTransform * translation;

		srcPoints[1] = Point2f(static_cast<float>(size.width), 0);
		srcPoints[2] = Point2f(0, static_cast<float>(size.height));
		srcPoints[3] = Point2f(static_cast<float>(size.width), static_cast<float>(size.height));

		perspectiveTransform(srcPoints, dstPoints, fullTransform);

		for (int j = 0; j < 4; ++j) {
			minX = min(minX, dstPoints[j].x);
			maxX = max(maxX, dstPoints[j].x);
			minY = min(minY, dstPoints[j].y);
			maxY = max(maxY, dstPoints[j].y);
		}

		fullTransforms[i] = fullTransform;
	}

	Size finalSize(static_cast<int>(maxX - minX), static_cast<int>(maxY - minY));
	Mat finalImage(finalSize, _images[0].type());
	Mat transformImage(finalSize, _images[0].type());
	Mat offset = Mat::eye(3, 3, CV_64F);
	Mat mask = Mat::zeros(finalSize, CV_8U);

	offset.at<double>(0, 2) = -minX;
	offset.at<double>(1, 2) = -minY;

	vector<Mat> warpedImages(_nbImages);
	vector<Mat> warpedMasks(_nbImages);

	for (int i = 0; i < _nbImages; ++i) {
		const Mat &img = getImage(i);
		const Size &size = img.size();
		Mat finalTransform = offset * fullTransforms[i];

		mask.rowRange(0, size.height).colRange(0, size.width).setTo(1);

		warpPerspective(img, warpedImages[i], finalTransform, finalSize, INTER_LINEAR);
		warpPerspective(mask, warpedMasks[i], finalTransform, finalSize, INTER_LINEAR);
		warpedImages[i].copyTo(finalImage, warpedMasks[i]);

		mask.setTo(0);
	}

	return finalImage;
}*/
