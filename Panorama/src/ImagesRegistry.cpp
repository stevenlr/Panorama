#include "ImagesRegistry.h"

#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

void ImagesRegistry::addImage(const Mat &img)
{
	int imageId = _images.size();
	ImageDescriptor descriptor;

	_images.push_back(img);

	descriptor.image = imageId;
	descriptor.width = img.size().width;
	descriptor.height = img.size().height;

	Ptr<FeatureDetector> featureDetector = FeatureDetector::create("SIFT");
	Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create("SIFT");
	Mat imageBW;

	cvtColor(img, imageBW, CV_RGB2GRAY);
	featureDetector->detect(imageBW, descriptor.keypoints);
	descriptorExtractor->compute(imageBW, descriptor.keypoints, descriptor.featureDescriptor);

	_descriptors.push_back(descriptor);
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
