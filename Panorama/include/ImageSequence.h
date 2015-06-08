#ifndef IMAGE_SEQUENCE_H_
#define IMAGE_SEQUENCE_H_

#include <vector>

#include "ImagesRegistry.h"

class ImageSequence {
public:
	ImageSequence() {}
	void addImage(int i, const ImagesRegistry &images);

private:
	ImageSequence(const ImageSequence &seq);
	ImageSequence &operator=(const ImageSequence &seq);

	std::vector<int> _keyFrames;
};

#endif
