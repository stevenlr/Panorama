#include "Scene.h"

#include <set>
#include <queue>
#include <algorithm>
#include <iostream>
#include <ctime>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
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
	_cameras.push_back(Camera());
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

int Scene::getRootNode() const
{
	int parent, node = 0;

	while ((parent = getParent(node)) != -1) {
		node = parent;
	}

	return node;
}

Mat_<double> Scene::computeError(const ImagesRegistry &images, const MatchGraph &matchGraph, int nbFeaturesTotal) const
{
	Mat_<double> error(Size(1, nbFeaturesTotal * 2));
	int errorId = 0;

	for (int i = 0; i < _nbImages; ++i) {
		Mat_<double> Hscene = _cameras[i].getH().inv();
		const vector<KeyPoint> ptsScene = images.getDescriptor(_images[i]).keypoints;

		for (int j = 0; j < _nbImages; ++j) {
			if (i == j) {
				continue;
			}

			const ImageMatchInfos &match = matchGraph.getImageMatchInfos(i, j);
			const vector<KeyPoint> ptsObject = images.getDescriptor(_images[j]).keypoints;
			Mat_<double> Hobj = _cameras[j].getH();
			int nbMatches = match.matches.size();

			for (int e = 0; e < nbMatches; ++e) {
				if (match.inliersMask[e]) {
					Point2d ptObject = ptsObject[match.matches[e].second].pt;
					Point2d ptScene = ptsScene[match.matches[e].first].pt;

					Mat_<double> m(Size(1, 3), CV_64F);

					m(0, 0) = ptObject.x + _cameras[j].ppx;
					m(1, 0) = ptObject.y + _cameras[j].ppy;
					m(2, 0) = 1;

					m = Hscene * Hobj * m;

					ptObject.x = m(0, 0) / m(2, 0) - _cameras[i].ppx;
					ptObject.y = m(1, 0) / m(2, 0) - _cameras[i].ppy;

					Point2d distance = ptObject - ptScene;

					error(errorId++, 0) = distance.x;
					error(errorId++, 0) = distance.y;

					cout << norm(distance) << endl;
				}
			}
		}
	}

	return error;
}

#define PARAM_ROTATION_X 0
#define PARAM_ROTATION_Y 1
#define PARAM_ROTATION_Z 2
#define PARAM_FOCAL_LENGTH 3

cv::Mat_<double> Scene::getErrorDerivative(int paramScene, int paramObject, bool firstAsDerivative, Point2d pointScene, Point2d pointObj) const
{
	Mat_<double> matPointObjDer = Mat::ones(Size(1, 3), CV_64F);
	Mat_<double> matPointObj = Mat::ones(Size(1, 3), CV_64F);
	int paramType = (firstAsDerivative) ? paramScene % 4 : paramObject % 4;
	int imgScene = paramScene / 4;
	int imgObject = paramObject / 4;

	matPointObjDer(0, 0) = pointObj.x;
	matPointObjDer(1, 0) = pointObj.y;

	matPointObj(0, 0) = pointObj.x;
	matPointObj(1, 0) = pointObj.y;

	if (firstAsDerivative) {
		if (paramType == PARAM_FOCAL_LENGTH) {
			Mat_<double> derK = Mat::zeros(Size(3, 3), CV_64F);

			derK(0, 0) = 1;
			derK(1, 1) = _cameras[imgScene].getAspectRatio();

			derK = -_cameras[imgScene].getK().inv() * derK * _cameras[imgScene].getK().inv();
			matPointObjDer = _cameras[imgScene].getR().inv() * derK * _cameras[imgObject].getH() * matPointObjDer;
		} else {
			Mat_<double> derR = Mat::zeros(Size(3, 3), CV_64F);

			if (paramType == PARAM_ROTATION_X) {
				derR(1, 2) = -1;
				derR(2, 1) = 1;
			} else if (paramType == PARAM_ROTATION_Y) {
				derR(2, 0) = -1;
				derR(0, 2) = 1;
			} else if (paramType == PARAM_ROTATION_Z) {
				derR(0, 1) = -1;
				derR(1, 0) = 1;
			}

			matPointObjDer = _cameras[imgScene].getK() * _cameras[imgScene].getR() * derR * _cameras[imgObject].getH().inv() * matPointObjDer;
		}
	} else {
		if (paramType == PARAM_FOCAL_LENGTH) {
			Mat_<double> derK = Mat::zeros(Size(3, 3), CV_64F);

			derK(0, 0) = 1;
			derK(1, 1) = _cameras[imgObject].getAspectRatio();

			derK = -_cameras[imgObject].getK().inv() * derK * _cameras[imgObject].getK().inv();
			matPointObjDer = _cameras[imgScene].getH() * _cameras[imgObject].getR().inv() * derK * matPointObjDer;
		} else {
			Mat_<double> derR = Mat::zeros(Size(3, 3), CV_64F);

			if (paramType == PARAM_ROTATION_X) {
				derR(1, 2) = -1;
				derR(2, 1) = 1;
			} else if (paramType == PARAM_ROTATION_Y) {
				derR(2, 0) = -1;
				derR(0, 2) = 1;
			} else if (paramType == PARAM_ROTATION_Z) {
				derR(0, 1) = -1;
				derR(1, 0) = 1;
			}

			derR = -_cameras[imgObject].getR().inv() * _cameras[imgObject].getR() * derR * _cameras[imgObject].getR().inv();
			matPointObjDer = _cameras[imgScene].getH() * derR * _cameras[imgObject].getK().inv() * matPointObjDer;
		}
	}

	matPointObj = _cameras[imgScene].getH().inv() * _cameras[imgObject].getH() * matPointObj;

	Mat_<double> homogeneousDer = Mat::zeros(Size(3, 2), CV_64F);

	homogeneousDer(0, 0) = 1 / matPointObj(2, 0);
	homogeneousDer(1, 1) = 1 / matPointObj(2, 0);
	homogeneousDer(0, 2) = -matPointObj(0, 0) / (matPointObj(2, 0) * matPointObj(2, 0));
	homogeneousDer(1, 2) = -matPointObj(1, 0) / (matPointObj(2, 0) * matPointObj(2, 0));

	return homogeneousDer * matPointObjDer * -1;
}

Mat_<double> Scene::getSingleError(int imgScene, int imgObj, Point2d pointScene, Point2d pointObj) const
{
	Mat_<double> m(Size(1, 3), CV_64F);

	m(0, 0) = pointObj.x + _cameras[imgObj].ppx;
	m(1, 0) = pointObj.y + _cameras[imgObj].ppy;
	m(2, 0) = 1;

	m = _cameras[imgScene].getH().inv() * _cameras[imgObj].getH() * m;

	pointObj.x = m(0, 0) / m(2, 0) - _cameras[imgScene].ppx;
	pointObj.y = m(1, 0) / m(2, 0) - _cameras[imgScene].ppy;

	Point2d distance = pointObj - pointScene;

	Mat_<double> pointMat(Size(1, 2), CV_64F);

	pointMat(0, 0) = distance.x;
	pointMat(1, 0) = distance.y;

	return pointMat;
}

void Scene::bundleAdjustment(const ImagesRegistry &images, const MatchGraph &matchGraph)
{
	for (int i = 0; i < _nbImages; ++i) {
		Size size = images.getImage(_images[i]).size();

		_cameras[i].focalLength = _estimatedFocalLength;
		_cameras[i].width = size.width;
		_cameras[i].height = size.height;
		_cameras[i].ppx = -size.width / 2;
		_cameras[i].ppy = -size.height / 2;
	}

	int rootNode = getRootNode();
	Mat_<double> K0 = _cameras[rootNode].getK();
	SVD svd;

	for (int i = 0; i < _nbImages; ++i) {
		if (i == rootNode) {
			continue;
		}

		svd(_cameras[i].getK().inv() * getFullTransform(i).inv() * K0, SVD::FULL_UV);

		Mat R = svd.u * svd.vt;

		if (determinant(R) < 0) {
			R *= -1;
		}

		Rodrigues(R, _cameras[rootNode].rotation);
	}

	int nbFeaturesTotal = 0;

	for (int i = 0; i < _nbImages; ++i) {
		for (int j = 0; j < _nbImages; ++j) {
			if (i == j) {
				continue;
			}

			nbFeaturesTotal += matchGraph.getImageMatchInfos(_images[i], _images[j]).nbInliers;
		}
	}

	Mat_<double> error = computeError(images, matchGraph, nbFeaturesTotal);
	Mat_<double> parameterDeviation = Mat::zeros(Size(_nbImages * 4, _nbImages * 4), CV_64F);

	for (int i = 0; i < _nbImages; ++i) {
		parameterDeviation(i * 4 + 0, i * 4 + 0) = PI / 16;
		parameterDeviation(i * 4 + 1, i * 4 + 1) = PI / 16;
		parameterDeviation(i * 4 + 2, i * 4 + 2) = PI / 16;
		parameterDeviation(i * 4 + 3, i * 4 + 3) = _estimatedFocalLength / 10;
	}

	double lambda = 1;
	Mat_<double> JtJ = Mat::zeros(Size(_nbImages * 4, _nbImages * 4), CV_64F);
	Mat_<double> Jtr = Mat::zeros(Size(1, _nbImages * 4), CV_64F);

	for (int i = 0; i < _nbImages * 4; ++i) {
		double sumJtr = 0;

		for (int j = 0; j < _nbImages * 4; ++j) {
			double sumJtJ = 0;
			const ImageMatchInfos &match = matchGraph.getImageMatchInfos(_images[i / 4], _images[j / 4]);
			int nbMatches = match.matches.size();

			for (int e = 0; e < nbMatches; ++e) {
				if (match.inliersMask[e]) {
					Point2d pointObj = images.getDescriptor(_images[j / 4]).keypoints[match.matches[e].second].pt;
					Point2d pointScene = images.getDescriptor(_images[i / 4]).keypoints[match.matches[e].first].pt;

					Mat_<double> errDerivative1 = getErrorDerivative(i, j, true, pointScene, pointObj);
					Mat_<double> errDerivative2 = getErrorDerivative(i, j, false, pointScene, pointObj);
					Mat_<double> termJtJ = errDerivative1.t() * errDerivative2;
					
					sumJtJ += termJtJ(0, 0);

					if (j % 4 == 0) {
						Mat_<double> singleError = getSingleError(i / 4, j / 4, pointScene, pointObj);
						Mat_<double> termJtr = errDerivative1.t() * singleError;

						sumJtr += termJtr(0, 0);
					}
				}
			}

			if (i == j) {
				sumJtJ += lambda / parameterDeviation(i, j);
			}

			JtJ(i, j) = sumJtJ;
		}

		Jtr(i, 0) = sumJtr;
	}

	Mat_<double> parameters = JtJ.inv() * Jtr;
}

#define WEIGHT_MAX 1000.0f

Mat Scene::composePanoramaSpherical(const ImagesRegistry &images, int projSizeX, int projSizeY)
{
	vector<Mat> warpedImages(_nbImages);
	vector<Mat> warpedMasks(_nbImages);
	vector<Mat> warpedWeights(_nbImages);
	vector<pair<Point2d, Point2d>> corners(_nbImages);
	Point finalMinCorner, finalMaxCorner;
	clock_t start;
	float elapsedTime;

	if (_nbImages < 2) {
		return Mat();
	}

	finalMinCorner.x = numeric_limits<int>::max();
	finalMinCorner.y = numeric_limits<int>::max();
	finalMaxCorner.x = numeric_limits<int>::min();
	finalMaxCorner.y = numeric_limits<int>::min();

	cout << "  Warping images";

	start = clock();
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

	elapsedTime = static_cast<float>(clock() - start) / _nbImages / CLOCKS_PER_SEC;
	cout << endl << "  Warping average: " << elapsedTime << "s" << endl;

	finalMinCorner.x = std::max(finalMinCorner.x, 0);
	finalMinCorner.y = std::max(finalMinCorner.y, 0);
	finalMaxCorner.x = std::min(finalMaxCorner.x, projSizeX - 1);
	finalMaxCorner.y = std::min(finalMaxCorner.y, projSizeY - 1);

	cout << finalMinCorner << endl;
	cout << finalMaxCorner << endl;

	Mat overlapIntensities(Size(_nbImages, _nbImages), CV_64F, Scalar(0));
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
	cout << "  Gain compensation average: " << elapsedTime << "s" << endl;

	/*cout << "  Building weight masks" << endl;

	start = clock();

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

	elapsedTime = static_cast<float>(clock() - start) / _nbImages / CLOCKS_PER_SEC;
	cout << "  Weight mask building average: " << elapsedTime << "s" << endl;

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

	start = clock();

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

	elapsedTime = static_cast<float>(clock() - start) / CLOCKS_PER_SEC;
	cout << "  Multiband blending total: " << elapsedTime << "s" << endl;*/

	start = clock();


	Size finalImageSize(finalMaxCorner.x - finalMinCorner.x, finalMaxCorner.y - finalMinCorner.y);
	Mat finalImage(finalImageSize, images.getImage(getImage(0)).type());
	Mat compositeImage(finalImage.size(), CV_32FC3, Scalar(0, 0, 0));
	
	TermCriteria criteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 10, 1.0);

	for (int y = 0; y < finalImageSize.height; ++y) {
		for (int x = 0; x < finalImageSize.width; ++x) {
			Mat labels, centers;
			int nbValues = 0;

			for (int i = 0; i < _nbImages; ++i) {
				if (warpedMasks[i].at<uchar>(y + finalMinCorner.y, x + finalMinCorner.x)) {
					nbValues++;
				}
			}

			if (nbValues == 0) {
				finalImage.at<Vec3b>(y, x)[0] = 0;
				finalImage.at<Vec3b>(y, x)[1] = 0;
				finalImage.at<Vec3b>(y, x)[2] = 0;
				continue;
			}

			Mat values(nbValues, 3, CV_32F);
			nbValues = 0;

			for (int i = 0; i < _nbImages; ++i) {
				if (warpedMasks[i].at<uchar>(y + finalMinCorner.y, x + finalMinCorner.x)) {
					values.at<float>(nbValues, 0) = warpedImages[i].at<Vec3b>(y + finalMinCorner.y, x + finalMinCorner.x)[0];
					values.at<float>(nbValues, 1) = warpedImages[i].at<Vec3b>(y + finalMinCorner.y, x + finalMinCorner.x)[1];
					values.at<float>(nbValues++, 2) = warpedImages[i].at<Vec3b>(y + finalMinCorner.y, x + finalMinCorner.x)[2];
				}
			}

			const int nbClusters = 3;

			if (nbValues < nbClusters) {
				finalImage.at<Vec3b>(y, x)[0] = values.at<float>(0, 0);
				finalImage.at<Vec3b>(y, x)[1] = values.at<float>(0, 1);
				finalImage.at<Vec3b>(y, x)[2] = values.at<float>(0, 2);
			} else {
				kmeans(values, nbClusters, labels, criteria, criteria.maxCount, KMEANS_RANDOM_CENTERS, centers);
				centers.convertTo(centers, CV_8U);

				int samplesCount[nbClusters];

				for (int i = 0; i < nbClusters; ++i) {
					samplesCount[i] = 0;
				}

				for (int i = 0; i < nbValues; ++i) {
					samplesCount[labels.at<int>(i, 0)]++;
				}

				int sampleCountMax = 0;
				int clusterMax = -1;

				for (int i = 0; i < nbClusters; ++i) {
					if (samplesCount[i] > sampleCountMax) {
						sampleCountMax = samplesCount[i];
						clusterMax = i;
					}
				}

				finalImage.at<Vec3b>(y, x)[0] = centers.at<uchar>(clusterMax, 0);
				finalImage.at<Vec3b>(y, x)[1] = centers.at<uchar>(clusterMax, 1);
				finalImage.at<Vec3b>(y, x)[2] = centers.at<uchar>(clusterMax, 2);
			}
		}
	}

	elapsedTime = static_cast<float>(clock() - start) / CLOCKS_PER_SEC;
	cout << "  kmeans total: " << elapsedTime << "s" << endl;

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
