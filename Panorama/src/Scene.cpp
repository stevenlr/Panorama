#include "Scene.h"

#include <set>
#include <queue>
#include <algorithm>
#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>

#include "ImageMatching.h"
#include "Constants.h"

using namespace std;
using namespace cv;

Scene::Scene(int nbImages)
{
	assert(nbImages > 0);

	_nbImages = nbImages;
	_images.resize(_nbImages);
	_imagesBW.resize(_nbImages);
	
	for (int i = 0; i < _nbImages; ++i) {
		_transform.push_back(Mat());
		_parent.push_back(-1);
	}
}

void Scene::setImage(int i, const Mat &img)
{
	assert(i >= 0 && i < _nbImages);

	Mat imgBW;

	_images[i] = img;
	cvtColor(img, imgBW, CV_RGB2GRAY);
	_imagesBW[i] = imgBW;
}

const Mat &Scene::getImage(int i) const
{
	assert(i >= 0 && i < _nbImages);

	return _images[i];
}

const cv::Mat &Scene::getImageBW(int i) const
{
	assert(i >= 0 && i < _nbImages);

	return _imagesBW[i];
}

int Scene::getNbImages() const
{
	return _nbImages;
}

int Scene::getParent(int image) const
{
	assert(image >= 0 && image < _nbImages);

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

Mat Scene::composePanoramaSpherical(int projSizeX, int projSizeY, double focalLength)
{
	vector<Mat> warpedImages(_nbImages);
	vector<Mat> warpedMasks(_nbImages);
	vector<Mat> warpedWeights(_nbImages);
	vector<pair<Point2d, Point2d>> corners(_nbImages);

	cout << "  Warping images";

	for (int i = 0; i < _nbImages; ++i) {
		Mat img = getImage(i);
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
				spacePoint.at<double>(0, 0) = sin(angleX) * cos(angleY) * focalLength;
				spacePoint.at<double>(1, 0) = sin(angleY) * focalLength;
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
		Mat weightNormal(size, CV_32F);

		for (int y = 0; y < size.height; ++y) {
			float *ptr = weightNormal.ptr<float>(y);

			for (int x = 0; x < size.width; ++x) {
				*ptr++ = getWeight(x, size.width) * getWeight(y, size.height) * WEIGHT_MAX;
			}
		}

		remap(maskNormal, warpedMasks[i], map, Mat(), INTER_LINEAR, BORDER_TRANSPARENT);
		remap(img, warpedImages[i], map, Mat(), INTER_LINEAR, BORDER_TRANSPARENT);
		remap(weightNormal, warpedWeights[i], map, Mat(), INTER_LINEAR, BORDER_TRANSPARENT);
		corners[i] = make_pair(minCorner, maxCorner);
	}

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

		for (int y = 0; y < projSizeY; ++y) {
			for (int i = 0; i < _nbImages; ++i) {
				ptrs[i] = warpedWeights[i].ptr<float>(y);
			}

			for (int x = 0; x < projSizeX; ++x) {
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

	const int nbBands = 4;
	vector<vector<Mat>> mbWeights(_nbImages);
	vector<vector<Mat>> mbBands(_nbImages);
	vector<vector<Mat>> mbImages(_nbImages);

	cout << "  Building frequency bands";

	{
		for (int i = 0; i < _nbImages; ++i) {
			float blurDeviation = 2.5;
			Mat mask1, mask3;

			cout << ".";

			warpedMasks[i].convertTo(mask1, CV_32F);
			mask1 *= WEIGHT_MAX;
			cvtColor(mask1, mask3, CV_GRAY2RGB);

			mbImages[i].resize(nbBands + 1);
			mbBands[i].resize(nbBands + 1);
			mbWeights[i].resize(nbBands + 1);

			warpedImages[i].convertTo(mbImages[i][0], CV_32FC3);
			mbWeights[i][0] = warpedWeights[i];

			for (int k = 1; k <= nbBands; ++k) {
				Mat maskBlurred1, maskBlurred3;

				GaussianBlur(mbImages[i][k - 1], mbImages[i][k], Size(0, 0), blurDeviation);
				GaussianBlur(mbWeights[i][k - 1], mbWeights[i][k], Size(0, 0), blurDeviation);
				GaussianBlur(mask1, maskBlurred1, Size(0, 0), blurDeviation);
				GaussianBlur(mask3, maskBlurred3, Size(0, 0), blurDeviation);

				multiply(mbImages[i][k], 2 - (maskBlurred3 / WEIGHT_MAX), mbImages[i][k]);
				multiply(mbWeights[i][k], 2 - (maskBlurred1 / WEIGHT_MAX), mbWeights[i][k]);

				mbBands[i][k] = mbImages[i][k - 1] - mbImages[i][k];

				blurDeviation *= sqrtf(2 * static_cast<float>(k) + 1);
			}

			mbBands[i][nbBands] = mbImages[i][nbBands];
		}
	}

	Mat finalImage(Size(projSizeX, projSizeY), getImage(0).type());
	Mat compositeImage(Size(projSizeX, projSizeY), CV_32FC3, Scalar(0, 0, 0));

	cout << endl << "  Compositing final image";

	for (int k = 1; k <= nbBands; ++k) {
		Mat img(finalImage.size(), CV_32FC3, Scalar(0, 0, 0));
		Mat sumWeights(finalImage.size(), CV_32FC3, Scalar(0, 0, 0));

		cout << ".";

		for (int i = 0; i < _nbImages; ++i) {
			Mat weight;

			cvtColor(mbWeights[i][k], weight, CV_GRAY2RGB);

			multiply(mbBands[i][k], weight / WEIGHT_MAX, mbBands[i][k]);
			add(img, mbBands[i][k], img, warpedMasks[i]);
			add(sumWeights, weight / WEIGHT_MAX, sumWeights, warpedMasks[i]);
		}

		divide(img, sumWeights, img);
		add(compositeImage, img, compositeImage);
	}

	cout << endl;
	compositeImage.convertTo(finalImage, CV_8UC3);

	return finalImage;
}

Mat Scene::composePanoramaPlanar()
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
}

bool Scene::checkCycle(int image) const
{
	set<int> stack;
	int current = image;

	stack.insert(image);

	while (current != -1) {
		current = getParent(current);

		if (stack.find(current) != stack.end()) {
			return true;
		}

		stack.insert(current);
	}

	return false;
}

void Scene::findSpanningTree(list<MatchGraphEdge> &matchGraphEdges, vector<vector<bool>> &matchSpanningTreeEdges)
{
	vector<int> connexComponents(_nbImages);
	bool fullyConnex = false;

	matchGraphEdges.sort(compareMatchGraphEdge);

	for (int i = 0; i < _nbImages; ++i) {
		connexComponents[i] = i;
		matchSpanningTreeEdges[i].resize(_nbImages);

		for (int j = 0; j < _nbImages; ++j) {
			matchSpanningTreeEdges[i][j] = false;
		}
	}

	while (!matchGraphEdges.empty() && !fullyConnex) {
		MatchGraphEdge elt(matchGraphEdges.front());
		int objectImage = elt.objectImage;
		int sceneImage = elt.sceneImage;

		matchGraphEdges.pop_front();

		if (connexComponents[objectImage] == connexComponents[sceneImage]) {
			continue;
		}

		fullyConnex = true;
		matchSpanningTreeEdges[objectImage][sceneImage] = true;
		matchSpanningTreeEdges[sceneImage][objectImage] = true;

		for (int i = 0; i < _nbImages; ++i) {
			if (connexComponents[i] == connexComponents[objectImage]) {
				connexComponents[i] = connexComponents[sceneImage];
			}

			if (connexComponents[i] != connexComponents[0]) {
				fullyConnex = false;
			}
		}
	}
}

void Scene::markNodeDepth(vector<int> &nodeDepth, vector<vector<bool>> &matchSpanningTreeEdges)
{
	for (int i = 0; i < _nbImages; ++i) {
		nodeDepth[i] = numeric_limits<int>::max();
	}

	for (int i = 0; i < _nbImages; ++i) {
		set<int> visited;
		queue<pair<int, int>> toVisit;
		int nbConnections = 0;

		for (int j = 0; j < _nbImages; ++j) {
			if (matchSpanningTreeEdges[i][j]) {
				nbConnections++;
			}
		}

		if (nbConnections != 1) {
			continue;
		}

		toVisit.push(make_pair(i, 0));
		
		while (!toVisit.empty()) {
			int current = toVisit.front().first;
			int depth = toVisit.front().second;

			nodeDepth[current] = min(nodeDepth[current], depth);
			visited.insert(current);

			for (int j = 0; j < _nbImages; ++j) {
				if (matchSpanningTreeEdges[current][j] && visited.find(j) == visited.end()) {
					toVisit.push(make_pair(j, depth + 1));
				}
			}

			toVisit.pop();
		}
	}
}

void Scene::makeFinalSceneTree(int treeCenter, map<pair<int, int>, ImageMatchInfos> &matchInfosMap,
							   vector<vector<bool>> &matchSpanningTreeEdges)
{
	set<int> visited;
	queue<pair<int, int>> toVisit;

	toVisit.push(make_pair(treeCenter, -1));
		
	while (!toVisit.empty()) {
		int current = toVisit.front().first;
		int parent = toVisit.front().second;

		visited.insert(current);
		setParent(current, parent);
			
		if (parent != -1) {
			setTransform(current, matchInfosMap[make_pair(current, parent)].homography);
		} else {
			setTransform(current, Mat::eye(Size(3, 3), CV_64F));
		}

		for (int j = 0; j < _nbImages; ++j) {
			if (matchSpanningTreeEdges[current][j] && visited.find(j) == visited.end()) {
				toVisit.push(make_pair(j, current));
			}
		}

		toVisit.pop();
	}
}

void Scene::makeSceneGraph(list<MatchGraphEdge> &matchGraphEdges, map<pair<int, int>, ImageMatchInfos> &matchInfosMap)
{
	vector<vector<bool>> matchSpanningTreeEdges(_nbImages);
	findSpanningTree(matchGraphEdges, matchSpanningTreeEdges);

	vector<int> nodeDepth(_nbImages);
	markNodeDepth(nodeDepth, matchSpanningTreeEdges);

	int treeCenter = max_element(nodeDepth.begin(), nodeDepth.end()) - nodeDepth.begin();
	makeFinalSceneTree(treeCenter, matchInfosMap, matchSpanningTreeEdges);
}
