#ifndef SCENE_H_
#define SCENE_H_

#include <vector>
#include <list>
#include <map>

#include <opencv2/core/core.hpp>

struct MatchGraphEdge;
struct ImageMatchInfos;

class Scene {
public:
	Scene(int nbImages);

	void setImage(int i, const cv::Mat &img);
	const cv::Mat &getImage(int i) const;
	const cv::Mat &getImageBW(int i) const;

	int getNbImages() const;
	int getParent(int image) const;
	void setParent(int image, int parent);
	void setTransform(int image, const cv::Mat &transform);
	const cv::Mat &getTransform(int image) const;

	cv::Mat composePanoramaPlanar();
	cv::Mat composePanoramaSpherical(int projSizeX, int projSizeY, double focalLength);

	cv::Mat getFullTransform(int image) const;
	bool checkCycle(int image) const;
	void makeSceneGraph(std::list<MatchGraphEdge> &matchGraphEdges, std::map<std::pair<int, int>, ImageMatchInfos> &matchInfosMap);

private:
	Scene(const Scene &);
	Scene &operator=(const Scene &);

	void findSpanningTree(std::list<MatchGraphEdge> &matchGraphEdges, std::vector<std::vector<bool>> &matchSpanningTreeEdges);
	void markNodeDepth(std::vector<int> &nodeDepth, std::vector<std::vector<bool>> &matchSpanningTreeEdges);
	void makeFinalSceneTree(int treeCenter, std::map<std::pair<int, int>, ImageMatchInfos> &matchInfosMap,
							std::vector<std::vector<bool>> &matchSpanningTreeEdges);

	int _nbImages;
	std::vector<cv::Mat> _images;
	std::vector<cv::Mat> _imagesBW;
	std::vector<int> _parent;
	std::vector<cv::Mat> _transform;
};

#endif
