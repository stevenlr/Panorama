#include "Scene.h"

#include <set>

#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

Scene::Scene(int nbImages)
{
	assert(nbImages > 0);

	_nbImages = nbImages;
	_images.resize(_nbImages);
	_imagesBW.resize(_nbImages);
	
	for (int i = 0; i < _nbImages; ++i) {
		_transform.push_back(Mat::eye(3, 3, CV_64F));
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

Mat Scene::composePanorama()
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
