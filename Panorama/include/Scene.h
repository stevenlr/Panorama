#ifndef SCENE_H_
#define SCENE_H_

#include <vector>

#include <opencv2/core/core.hpp>

class Scene {
public:
	Scene(int nbImages);

	Scene(const Scene &) = delete;
	Scene &operator=(const Scene &) = delete;

	void setImage(int i, const cv::Mat &img);
	const cv::Mat &getImage(int i) const;
	const cv::Mat &getImageBW(int i) const;

	int getParent(int image) const;
	void setParent(int image, int parent);
	void setTransform(int image, const cv::Mat &transform);
	cv::Mat composePanorama();
	cv::Mat getFullTransform(int image) const;

private:
	int _nbImages;
	std::vector<cv::Mat> _images;
	std::vector<cv::Mat> _imagesBW;
	std::vector<int> _parent;
	std::vector<cv::Mat> _transform;
};

#endif
