#ifndef SCENE_H_
#define SCENE_H_

#include <vector>
#include <map>

#include <opencv2/core/core.hpp>

#include "ImagesRegistry.h"
#include "Camera.h"

struct MatchGraphEdge;
struct ImageMatchInfos;

class Scene {
public:
	Scene();

	void setEstimatedFocalLength(double f);
	void addImage(int id);
	int getNbImages() const;
	int getImage(int id) const;
	int getParent(int image) const;
	int getIdInScene(int image) const;
	void setParent(int image, int parent);
	void setTransform(int image, const cv::Mat &transform);
	const cv::Mat &getTransform(int image) const;
	int getRootNode() const;

	//cv::Mat composePanoramaPlanar();
	void bundleAdjustment(const ImagesRegistry &images);
	cv::Mat composePanoramaSpherical(const ImagesRegistry &images, int projSizeX, int projSizeY);

	cv::Mat getFullTransform(int image) const;

private:
	int _nbImages;
	double _estimatedFocalLength;
	std::map<int, int> _reverseIds;
	std::vector<int> _images;
	std::vector<int> _parent;
	std::vector<cv::Mat> _transform;
	std::vector<Camera> _cameras;
};

#endif
