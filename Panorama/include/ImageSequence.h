#ifndef IMAGE_SEQUENCE_H_
#define IMAGE_SEQUENCE_H_

#include <vector>

#include <opencv2/core/core.hpp>

#include "ImagesRegistry.h"
#include "Scene.h"

class ImageSequence {
public:
	ImageSequence() { _nbFrames = 0; }
	void addImage(int i, const ImagesRegistry &images);
	int getNbKeyframes() const;
	int getKeyFrame(int i) const;
	double estimateFocalLength();
	void addIntermediateFramesToScene(Scene &scene);

private:
	ImageSequence(const ImageSequence &seq);
	ImageSequence &operator=(const ImageSequence &seq);

	std::vector<int> _keyFrames;
	std::vector<double> _focalLengths;
	std::vector<cv::Mat> _homographies;
	int _nbFrames;
};

#endif
