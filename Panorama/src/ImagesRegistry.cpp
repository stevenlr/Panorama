#include "ImagesRegistry.h"

#include <iostream>
#include <thread>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

bool ImagesRegistry::addImage(const string &filename)
{
	int imageId = _images.size();
	ImageDescriptor descriptor;
	Mat img = imread(filename);

	if (!img.data) {
		return false;
	}

	_images.push_back(img);
	descriptor.image = imageId;
	descriptor.width = img.size().width;
	descriptor.height = img.size().height;
	_descriptors.push_back(descriptor);

	return true;
}

int ImagesRegistry::getNbImages() const
{
	return _images.size();
}

const ImageDescriptor &ImagesRegistry::getDescriptor(int id) const
{
	assert(id >= 0 && id < _images.size());

	return _descriptors[id];
}

const cv::Mat &ImagesRegistry::getImage(int id) const
{
	assert(id >= 0 && id < _images.size());

	return _images[id];
}

namespace {
}

void ImagesRegistry::taskExtractFeatures(queue<int> &tasks)
{
	Ptr<FeatureDetector> featureDetector = FeatureDetector::create("SIFT");
	Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create("SIFT");

	while (!tasks.empty()) {
		int task = tasks.front();
		tasks.pop();

		_printMutex.lock();
		cout << "\rExtracting image features " << (++_imagesDone) << "/" << _images.size() << flush;
		_printMutex.unlock();

		const Mat &img = _images[task];
		ImageDescriptor &descriptor = _descriptors[task];
		Mat imageBW;

		cvtColor(img, imageBW, CV_RGB2GRAY);
		featureDetector->detect(imageBW, descriptor.keypoints);
		descriptorExtractor->compute(imageBW, descriptor.keypoints, descriptor.featureDescriptor);
	}
}

void ImagesRegistry::extractFeatures()
{
	int nbThreads = thread::hardware_concurrency();
	vector<thread> threads(nbThreads);
	vector<queue<int>> taskQueues(nbThreads);

	_descriptors.resize(_images.size());
	_imagesDone = 0;

	for (int i = 0; i < _images.size(); ++i) {
		taskQueues[i % nbThreads].push(i);
	}

	for (int i = 0; i < nbThreads; ++i) {
		new(&threads[i]) thread(&ImagesRegistry::taskExtractFeatures, std::ref(*this), taskQueues[i]);
	}
	for (int i = 0; i < nbThreads; ++i) {
		threads[i].join();
	}

	cout << endl;
}
