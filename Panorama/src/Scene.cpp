#include "Scene.h"

#include <set>
#include <queue>
#include <algorithm>

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
	return _images.size();
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

Mat Scene::composePanoramaSpherical(int projSizeX, int projSizeY, double focalLength)
{
	int nbImages = getNbImages();
	Mat finalImage(Size(projSizeX, projSizeY), getImage(0).type());

	for (int i = 0; i < nbImages; ++i) {
		Mat img = getImage(i);
		Mat map(Size(projSizeX, projSizeY), CV_32FC2, Scalar(-1, -1));
		Mat homography = getFullTransform(i).clone();

		Mat translation = Mat::eye(Size(3, 3), CV_64F);

		translation.at<double>(0, 2) = -img.size().width / 2;
		translation.at<double>(1, 2) = -img.size().height / 2;

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

				map.at<Vec2f>(y, x)[0] = static_cast<float>(projX);
				map.at<Vec2f>(y, x)[1] = static_cast<float>(projY);
			}
		}

		Mat maskNormal = Mat::ones(img.size(), CV_8U);
		Mat maskSpherical(Size(projSizeX, projSizeY), CV_8U, Scalar(0));
		Mat img2(Size(projSizeX, projSizeY), CV_8UC3, Scalar(0, 0, 0));

		remap(maskNormal, maskSpherical, map, Mat(), INTER_LINEAR, BORDER_TRANSPARENT);
		remap(img, img2, map, Mat(), INTER_LINEAR, BORDER_TRANSPARENT);
		img2.copyTo(finalImage, maskSpherical);
	}

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

	for (int i = 0; i < _nbImages; ++i) {
		const Mat &img = getImage(i);
		const Size &size = img.size();
		Mat finalTransform = offset * fullTransforms[i];

		mask.rowRange(0, size.height).colRange(0, size.width).setTo(1);

		warpPerspective(img, transformImage, finalTransform, finalSize, INTER_LINEAR);
		warpPerspective(mask, mask, finalTransform, finalSize, INTER_LINEAR);
		transformImage.copyTo(finalImage, mask);

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
	int nbImages = getNbImages();
	vector<int> connexComponents(nbImages);
	bool fullyConnex = false;

	matchGraphEdges.sort(compareMatchGraphEdge);

	for (int i = 0; i < nbImages; ++i) {
		connexComponents[i] = i;
		matchSpanningTreeEdges[i].resize(nbImages);

		for (int j = 0; j < nbImages; ++j) {
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

		for (int i = 0; i < nbImages; ++i) {
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
	int nbImages = getNbImages();

	for (int i = 0; i < nbImages; ++i) {
		nodeDepth[i] = numeric_limits<int>::max();
	}

	for (int i = 0; i < nbImages; ++i) {
		set<int> visited;
		queue<pair<int, int>> toVisit;
		int nbConnections = 0;

		for (int j = 0; j < nbImages; ++j) {
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

			for (int j = 0; j < nbImages; ++j) {
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
	int nbImages = getNbImages();
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

		for (int j = 0; j < nbImages; ++j) {
			if (matchSpanningTreeEdges[current][j] && visited.find(j) == visited.end()) {
				toVisit.push(make_pair(j, current));
			}
		}

		toVisit.pop();
	}
}

void Scene::makeSceneGraph(list<MatchGraphEdge> &matchGraphEdges, map<pair<int, int>, ImageMatchInfos> &matchInfosMap)
{
	int nbImages = getNbImages();

	vector<vector<bool>> matchSpanningTreeEdges(nbImages);
	findSpanningTree(matchGraphEdges, matchSpanningTreeEdges);

	vector<int> nodeDepth(nbImages);
	markNodeDepth(nodeDepth, matchSpanningTreeEdges);

	int treeCenter = max_element(nodeDepth.begin(), nodeDepth.end()) - nodeDepth.begin();
	makeFinalSceneTree(treeCenter, matchInfosMap, matchSpanningTreeEdges);
}
