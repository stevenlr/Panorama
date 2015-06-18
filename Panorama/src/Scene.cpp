#include "Scene.h"

#include <set>
#include <queue>
#include <algorithm>
#include <iostream>
#include <ctime>
#include <iomanip>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

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
	//vector<Mat> warpedWeights(_nbImages);
	vector<pair<Point2d, Point2d>> corners(_nbImages);
	Point finalMinCorner(numeric_limits<int>::max(), numeric_limits<int>::max());
	Point finalMaxCorner(numeric_limits<int>::min(), numeric_limits<int>::min());
	clock_t start;
	float elapsedTime;

	if (_nbImages < 2) {
		return Mat();
	}

	start = clock();
	for (int i = 0; i < _nbImages; ++i) {
		Mat img = images.getImage(getImage(i));
		Size size = img.size();
		Mat homography = getFullTransform(i).clone();
		Point2i minCorner(numeric_limits<int>::max(), numeric_limits<int>::max());
		Point2i maxCorner(numeric_limits<int>::min(), numeric_limits<int>::min());
		Mat translation = Mat::eye(Size(3, 3), CV_64F);

		if (homography.size() != Size(3, 3)) {
			continue;
		}

		cout << "\r  Warping images " << (i + 1) << "/" << _nbImages << flush;

		translation.at<double>(0, 2) = -size.width / 2;
		translation.at<double>(1, 2) = -size.height / 2;
		homography = homography * translation;

		for (int y = 0; y < size.height; ++y) {
			for (int x = 0; x < size.width; ++x) {
				if (y != 0 && x != 0 && y != size.height - 1 && x != size.width - 1) {
					continue;
				}

				Mat_<double> point = Mat_<double>::ones(Size(1, 3));

				point(0, 0) = x;
				point(1, 0) = y;

				point = homography * point;

				point(0, 0) /= _estimatedFocalLength;
				point(1, 0) /= _estimatedFocalLength;

				double angleX = atan2(point(0, 0), point(2, 0));
				double angleY = atan2(point(1, 0), sqrt(point(0, 0) * point(0, 0) + point(2, 0) * point(2, 0)));

				int px = static_cast<int>((angleX / PI + 0.5) * projSizeX);
				int py = static_cast<int>((angleY * 2 / PI + 0.5) * projSizeY);

				minCorner.x = std::min(minCorner.x, px);
				minCorner.y = std::min(minCorner.y, py);
				maxCorner.x = std::max(maxCorner.x, px);
				maxCorner.y = std::max(maxCorner.y, py);
			}
		}

		Mat invHomography = homography.inv();
		Mat map(Size(maxCorner.x - minCorner.x + 1, maxCorner.y - minCorner.y + 1), CV_32FC2, Scalar(-1, -1));

		for (int x = minCorner.x; x <= maxCorner.x; ++x) {
			double angleX = ((double) x / projSizeX - 0.5) * PI;

			for (int y = minCorner.y; y <= maxCorner.y; ++y) {
				double angleY = ((double) y / projSizeY - 0.5) * PI / 2;

				Mat spacePoint = Mat::zeros(Size(1, 3), CV_64F);
				spacePoint.at<double>(0, 0) = sin(angleX) * cos(angleY) * _estimatedFocalLength;
				spacePoint.at<double>(1, 0) = sin(angleY) * _estimatedFocalLength;
				spacePoint.at<double>(2, 0) = cos(angleX) * cos(angleY);

				Mat transformedPoint = invHomography * spacePoint;
				double projX = transformedPoint.at<double>(0, 0) / transformedPoint.at<double>(2, 0);
				double projY = transformedPoint.at<double>(1, 0) / transformedPoint.at<double>(2, 0);

				if (projX < 0 || projX >= size.width || projY < 0 || projY >= size.height) {
					continue;
				}

				map.at<Vec2f>(y - minCorner.y, x - minCorner.x)[0] = static_cast<float>(projX);
				map.at<Vec2f>(y - minCorner.y, x - minCorner.x)[1] = static_cast<float>(projY);
			}
		}
		
		Mat maskNormal(size, CV_8U, Scalar(255));

		remap(maskNormal, warpedMasks[i], map, Mat(), INTER_LINEAR, BORDER_CONSTANT);
		remap(img, warpedImages[i], map, Mat(), INTER_LINEAR, BORDER_CONSTANT);
		corners[i] = make_pair(minCorner, maxCorner);

		//distanceTransform(warpedMasks[i], warpedWeights[i], CV_DIST_L1, 3);

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

	start = clock();

	Size finalImageSize(finalMaxCorner.x - finalMinCorner.x + 1, finalMaxCorner.y - finalMinCorner.y + 1);
	Mat finalImage(finalImageSize, images.getImage(getImage(0)).type());
	Mat_<float> stdDevImage(finalImageSize);
	Mat compositeImage(finalImage.size(), CV_32FC3, Scalar(0, 0, 0));
	
	TermCriteria criteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 10, 1.0);

	for (int y = 0; y < finalImageSize.height; ++y) {
		cout << "\r  Extracting background " << static_cast<int>(static_cast<float>(y) / finalImageSize.height * 100) << "%" << flush;

		for (int x = 0; x < finalImageSize.width; ++x) {
			Mat labels, centers;
			int nbValues = 0;
			int px = x + finalMinCorner.x;
			int py = y + finalMinCorner.y;

			for (int i = 0; i < _nbImages; ++i) {
				int ix = px - corners[i].first.x;
				int iy = py - corners[i].first.y;

				if (ix < 0 || iy < 0 || ix >= warpedMasks[i].size().width || iy >= warpedMasks[i].size().height) {
					continue;
				}

				if (warpedMasks[i].at<uchar>(iy, ix)) {
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
			float meanR = 0, meanG = 0, meanB = 0;
			nbValues = 0;

			for (int i = 0; i < _nbImages; ++i) {
				int ix = px - corners[i].first.x;
				int iy = py - corners[i].first.y;

				if (ix < 0 || iy < 0 || ix >= warpedMasks[i].size().width || iy >= warpedMasks[i].size().height) {
					continue;
				}

				if (warpedMasks[i].at<uchar>(iy, ix)) {
					meanR += (values.at<float>(nbValues, 0) = warpedImages[i].at<Vec3b>(iy, ix)[0]);
					meanG += (values.at<float>(nbValues, 1) = warpedImages[i].at<Vec3b>(iy, ix)[1]);
					meanB += (values.at<float>(nbValues++, 2) = warpedImages[i].at<Vec3b>(iy, ix)[2]);
				}
			}

			meanR /= nbValues;
			meanG /= nbValues;
			meanB /= nbValues;

			float stdDev = 0;

			for (int i = 0; i < nbValues; ++i) {
				float diff = std::max(std::max(std::abs(meanR - values.at<float>(i, 0)),
					std::abs(meanG - values.at<float>(i, 1))),
					std::abs(meanB - values.at<float>(i, 2)));

				stdDev += diff * diff;
			}

			stdDev = std::sqrtf(stdDev / nbValues);

			const int nbClusters = 3;
			const float minStdDev = 35;

			if (stdDev > minStdDev && nbValues >= nbClusters) {
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

				meanR = centers.at<uchar>(clusterMax, 0);
				meanG = centers.at<uchar>(clusterMax, 1);
				meanB = centers.at<uchar>(clusterMax, 2);
				stdDev = 0;

				for (int i = 0; i < nbValues; ++i) {
					if (labels.at<int>(i, 0) == clusterMax) {
						float diff = std::max(std::max(std::abs(meanR - values.at<float>(i, 0)),
							std::abs(meanG - values.at<float>(i, 1))),
							std::abs(meanB - values.at<float>(i, 2)));

						stdDev += diff * diff;
					}
				}

				stdDev = std::sqrtf(stdDev / nbValues);
			}

			finalImage.at<Vec3b>(y, x)[0] = meanR;
			finalImage.at<Vec3b>(y, x)[1] = meanG;
			finalImage.at<Vec3b>(y, x)[2] = meanB;
			stdDevImage(y, x) = stdDev;
		}
	}

	cout << endl;

	for (int interestImage = 0; interestImage < _nbImages; ++interestImage) {
		Mat baseImage = images.getImage(getImage(interestImage));
		const Size &size = baseImage.size();
		Mat_<Vec2f> unwarp(baseImage.size());
		Mat homography = getFullTransform(interestImage).clone();
		Mat translation = Mat::eye(Size(3, 3), CV_64F);

		translation.at<double>(0, 2) = -size.width / 2;
		translation.at<double>(1, 2) = -size.height / 2;
		homography = homography * translation;

		unwarp.setTo(Scalar(-1, -1));

		cout << "\r  Extracting foreground " << (interestImage + 1) << "/" << _nbImages << flush;

		for (int y = 0; y < size.height; ++y) {
			for (int x = 0; x < size.width; ++x) {
				Mat_<double> point = Mat_<double>::ones(Size(1, 3));

				point(0, 0) = x;
				point(1, 0) = y;

				point = homography * point;

				point(0, 0) /= _estimatedFocalLength;
				point(1, 0) /= _estimatedFocalLength;

				double angleX = atan2(point(0, 0), point(2, 0));
				double angleY = atan2(point(1, 0), sqrt(point(0, 0) * point(0, 0) + point(2, 0) * point(2, 0)));

				unwarp(y, x)[0] = ((angleX / PI + 0.5) * projSizeX) - finalMinCorner.x;
				unwarp(y, x)[1] = ((angleY * 2 / PI + 0.5) * projSizeY) - finalMinCorner.y;
			}
		}

		Mat unwarpedBackground, difference, cleaned, unwarpedStdDev;
		Mat stdDev, mean;
		vector<Mat> channels(3);
		double thresholdValue;

		remap(finalImage, unwarpedBackground, unwarp, Mat(), INTER_LINEAR, BORDER_CONSTANT);
		remap(stdDevImage, unwarpedStdDev, unwarp, Mat(), INTER_LINEAR, BORDER_CONSTANT);
		GaussianBlur(baseImage, baseImage, Size(0, 0), 1, 0, BORDER_REPLICATE);
		GaussianBlur(unwarpedBackground, unwarpedBackground, Size(0, 0), 1, 0, BORDER_REPLICATE);
		absdiff(unwarpedBackground, baseImage, difference);
		split(difference, channels);
		difference = max(channels[2], max(channels[1], channels[0]));

		Mat_<uchar> thresholded(size);

		thresholded.setTo(0);

		for (int y = 0; y < size.height; ++y) {
			uchar *thresholdedPtr = thresholded.ptr(y);
			float *stdDevPtr = unwarpedStdDev.ptr<float>(y);
			uchar *differencePtr = difference.ptr<uchar>(y);

			for (int x = 0; x < size.width; ++x) {
				if (*differencePtr++ > *stdDevPtr++ * 2) {
					*thresholdedPtr = 255;
				}

				thresholdedPtr++;
			}
		}

		int closingRadius = 2;
		Mat element = getStructuringElement(MORPH_RECT, Size(closingRadius * 2 + 1, closingRadius * 2 + 1), Point(closingRadius, closingRadius));

		erode(thresholded, cleaned, element);

		Mat m0, m1;

		element = getStructuringElement(MORPH_RECT, Size(3, 3), Point(1, 1));
		m1 = cleaned.clone();

		do {
			m0 = m1.clone();
			dilate(m0, m1, element);
			cv::min(m1, thresholded, m1);
		} while (countNonZero(abs(m0 - m1)) != 0);

		thresholded = m1;

		stringstream sstr;

		sstr << "output_difference_" << setfill('0') << setw(4) << (interestImage + 1) << ".jpg";
		imwrite(sstr.str(), thresholded);
	}

	elapsedTime = static_cast<float>(clock() - start) / CLOCKS_PER_SEC;
	cout << endl << "  kmeans total: " << elapsedTime << "s" << endl;

	return finalImage;
}
