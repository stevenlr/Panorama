#include "Scene.h"

#include <set>
#include <queue>
#include <algorithm>
#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/stitching/detail/blenders.hpp>

#include "Constants.h"

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

Mat Scene::composePanoramaSpherical(const ImagesRegistry &images, int projSizeX, int projSizeY)
{
	vector<Mat> warpedImages(_nbImages);
	vector<Mat> warpedMasks(_nbImages);
	vector<Mat> warpedWeights(_nbImages);
	vector<pair<Point2d, Point2d>> corners(_nbImages);
	Point finalMinCorner, finalMaxCorner;

	if (_nbImages < 2) {
		return Mat();
	}

	finalMinCorner.x = numeric_limits<int>::max();
	finalMinCorner.y = numeric_limits<int>::max();
	finalMaxCorner.x = numeric_limits<int>::min();
	finalMaxCorner.y = numeric_limits<int>::min();

	cout << "  Warping images";

	for (int i = 0; i < _nbImages; ++i) {
		Mat img = images.getImage(getImage(i));
		Size size = img.size();
		Mat map(Size(projSizeX, projSizeY), CV_32FC2, Scalar(-1, -1));
		Mat homography = getFullTransform(i).clone();
		Point2i minCorner(numeric_limits<int>::max(), numeric_limits<int>::max());
		Point2i maxCorner(numeric_limits<int>::min(), numeric_limits<int>::min());
		Mat translation = Mat::eye(Size(3, 3), CV_64F);

		if (homography.size() != Size(3, 3)) {
			continue;
		}

		cout << ".";

		translation.at<double>(0, 2) = -size.width / 2;
		translation.at<double>(1, 2) = -size.height / 2;
		homography = homography * translation;

		Mat invHomography = homography.inv();

		for (int x = 0; x < projSizeX; ++x) {
			double angleX = ((double) x / projSizeX - 0.5) * PI;

			for (int y = 0; y < projSizeY; ++y) {
				double angleY = ((double) y / projSizeY - 0.5) * PI / 2;

				Mat spacePoint = Mat::zeros(Size(1, 3), CV_64F);
				spacePoint.at<double>(0, 0) = sin(angleX) * cos(angleY) * _estimatedFocalLength;
				spacePoint.at<double>(1, 0) = sin(angleY) * _estimatedFocalLength;
				spacePoint.at<double>(2, 0) = cos(angleX) * cos(angleY);

				Mat transformedPoint = invHomography * spacePoint;
				double projX = transformedPoint.at<double>(0, 0) / transformedPoint.at<double>(2, 0);
				double projY = transformedPoint.at<double>(1, 0) / transformedPoint.at<double>(2, 0);

				if (projX >= 0 && projX < size.width && projY >= 0 && projY < size.height) {
					minCorner.x = std::min(minCorner.x, x);
					minCorner.y = std::min(minCorner.y, y);
					maxCorner.x = std::max(maxCorner.x, x);
					maxCorner.y = std::max(maxCorner.y, y);
				}

				map.at<Vec2f>(y, x)[0] = static_cast<float>(projX);
				map.at<Vec2f>(y, x)[1] = static_cast<float>(projY);
			}
		}

		Mat maskNormal = Mat::ones(size, CV_8U);

		remap(maskNormal, warpedMasks[i], map, Mat(), INTER_LINEAR, BORDER_TRANSPARENT);
		remap(img, warpedImages[i], map, Mat(), INTER_LINEAR, BORDER_TRANSPARENT);
		corners[i] = make_pair(minCorner, maxCorner);

		distanceTransform(warpedMasks[i], warpedWeights[i], CV_DIST_L1, 3);

		finalMinCorner.x = std::min(finalMinCorner.x, minCorner.x);
		finalMinCorner.y = std::min(finalMinCorner.y, minCorner.y);
		finalMaxCorner.x = std::max(finalMaxCorner.x, maxCorner.x);
		finalMaxCorner.y = std::max(finalMaxCorner.y, maxCorner.y);
	}

	finalMinCorner.x = std::max(finalMinCorner.x, 0);
	finalMinCorner.y = std::max(finalMinCorner.y, 0);
	finalMaxCorner.x = std::min(finalMaxCorner.x, projSizeX - 1);
	finalMaxCorner.y = std::min(finalMaxCorner.y, projSizeY - 1);

	Mat overlapIntensities(Size(_nbImages, _nbImages), CV_64F, Scalar(0));
	Mat overlapSizes(Size(_nbImages, _nbImages), CV_32S, Scalar(0));

	cout << endl << "  Compensating exposure" << endl;

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

	cout << "  Building weight masks" << endl;

	{
		vector<float *> ptrs(_nbImages);

		for (int y = finalMinCorner.y; y <= finalMaxCorner.y; ++y) {
			for (int i = 0; i < _nbImages; ++i) {
				ptrs[i] = warpedWeights[i].ptr<float>(y) + finalMinCorner.x;
			}

			for (int x = finalMinCorner.x; x <= finalMaxCorner.x; ++x) {
				float maxWeight = 0;
				int maxWeightImage = -1;

				for (int i = 0; i < _nbImages; ++i) {
					float weight = *ptrs[i];

					if (weight > maxWeight) {
						maxWeight = weight;

						if (maxWeightImage != -1) {
							*ptrs[maxWeightImage] = 0;
						}

						*ptrs[i] = WEIGHT_MAX;
						maxWeightImage = i;
					} else {
						*ptrs[i] = 0;
					}
				}

				for (int i = 0; i < _nbImages; ++i) {
					++ptrs[i];
				}
			}
		}
	}

	const int nbBands = 5;
	Mat mbWeight, mbRgbWeight;
	Mat mbBand;
	Mat mbImage, mbNextImage;
	vector<Mat> mbSumWeight(nbBands);
	vector<Mat> mbSumImage(nbBands);
	Size finalImageSize(finalMaxCorner.x - finalMinCorner.x, finalMaxCorner.y - finalMinCorner.y);

	for (int i = 0; i < nbBands; ++i) {
		mbSumWeight[i].create(finalImageSize, CV_32F);
		mbSumImage[i].create(finalImageSize, CV_32FC3);

		mbSumWeight[i].setTo(0);
		mbSumImage[i].setTo(0);
	}

	cout << "  Building frequency bands";

	for (int i = 0; i < _nbImages; ++i) {
		float blurDeviation = 10;

		cout << ".";

		warpedImages[i].colRange(finalMinCorner.x, finalMaxCorner.x)
						.rowRange(finalMinCorner.y, finalMaxCorner.y)
						.copyTo(mbImage);
		mbImage.convertTo(mbImage, CV_32FC3);

		warpedWeights[i].colRange(finalMinCorner.x, finalMaxCorner.x)
						.rowRange(finalMinCorner.y, finalMaxCorner.y)
						.copyTo(mbWeight);
			
		warpedMasks[i] = warpedMasks[i].colRange(finalMinCorner.x, finalMaxCorner.x)
									   .rowRange(finalMinCorner.y, finalMaxCorner.y);

		for (int k = 0; k < nbBands; ++k) {
			GaussianBlur(mbImage, mbNextImage, Size(0, 0), blurDeviation);
			GaussianBlur(mbWeight, mbWeight, Size(0, 0), blurDeviation);

			mbBand = mbImage - mbNextImage;

			cvtColor(mbWeight, mbRgbWeight, CV_GRAY2RGB);

			if (k != nbBands - 1) {
				multiply(mbBand, mbRgbWeight / WEIGHT_MAX, mbBand);
			} else {
				multiply(mbNextImage, mbRgbWeight / WEIGHT_MAX, mbBand);
			}

			add(mbSumImage[k], mbBand, mbSumImage[k], warpedMasks[i]);
			add(mbSumWeight[k], mbWeight, mbSumWeight[k], warpedMasks[i]);

			blurDeviation /= sqrtf(2 * static_cast<float>(k) + 1);
			mbImage = mbNextImage;
		}
	}

	cout << endl << "  Compositing final image" << endl;

	Mat finalImage(finalImageSize, images.getImage(getImage(0)).type());
	Mat compositeImage(finalImage.size(), CV_32FC3, Scalar(0, 0, 0));
	Mat weightRgb;

	for (int k = 0; k < nbBands; ++k) {
		cvtColor(mbSumWeight[k], weightRgb, CV_GRAY2RGB);
		divide(mbSumImage[k], weightRgb / WEIGHT_MAX, mbSumImage[k]);
		add(compositeImage, mbSumImage[k], compositeImage);
	}

	compositeImage.convertTo(finalImage, CV_8UC3);

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
