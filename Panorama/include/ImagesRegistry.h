#ifndef IMAGES_REGISTRY_H_
#define IMAGES_REGISTRY_H_

#include <vector>

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
	void addImage(const Mat &img);
	int getNbImages() const;
	const ImageDescriptor &getDescriptor(int id) const;
private:
	std::vector<cv::Mat> _images;
	std::vector<ImageDescriptor> _descriptors;
};

#endif
