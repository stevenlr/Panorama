#ifndef SCENE_H_
#define SCENE_H_

#include <vector>
#include <map>

#include <opencv2/core/core.hpp>

#include "MatchGraph.h"
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

	void bundleAdjustment(const ImagesRegistry &images, const MatchGraph &matchGraph);
	cv::Mat composePanorama(const ImagesRegistry &images);

	cv::Mat getFullTransform(int image) const;

private:
	cv::Mat_<double> computeError(const ImagesRegistry &images, const MatchGraph &matchGraph, int nbFeaturesTotal) const;
	cv::Mat_<double> getErrorDerivative(int paramScene, int paramObject, bool firstAsDerivative, cv::Point2d pointScene, cv::Point2d pointObj) const;
	cv::Mat_<double> getSingleError(int imgScene, int imgObj, cv::Point2d pointScene, cv::Point2d pointObj) const;

	int _nbImages;
	double _estimatedFocalLength;
	std::map<int, int> _reverseIds;
	std::vector<int> _images;
	std::vector<int> _parent;
	std::vector<cv::Mat> _transform;
	std::vector<Camera> _cameras;
};

#endif
