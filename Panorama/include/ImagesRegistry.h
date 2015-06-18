#ifndef IMAGES_REGISTRY_H_
#define IMAGES_REGISTRY_H_

#include <vector>
#include <mutex>
#include <queue>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

struct ImageDescriptor {
	int image;
	int width;
	int height;
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat featureDescriptor;
};

class ImagesRegistry {
public:
	bool addImage(const std::string &filename);
	int getNbImages() const;
	const ImageDescriptor &getDescriptor(int id) const;
	const cv::Mat &getImage(int id) const;
	void extractFeatures();

private:
	void taskExtractFeatures(std::queue<int> &tasks);

	std::vector<cv::Mat> _images;
	std::vector<ImageDescriptor> _descriptors;
	std::mutex _printMutex;
	int _imagesDone;
};

#endif
