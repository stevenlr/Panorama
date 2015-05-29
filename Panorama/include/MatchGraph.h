#ifndef MATCH_GRAPH_H_
#define MATCH_GRAPH_H_

#include <vector>
#include <list>

#include "ImagesRegistry.h"
#include "Scene.h"

struct ImageMatchInfos {
	std::vector<std::pair<int, int>> matches;
	cv::Mat homography;
	double confidence;
	int nbInliers;
	int nbOverlaps;
	std::vector<uchar> inliersMask;

	ImageMatchInfos();
	ImageMatchInfos(const ImageMatchInfos &infos);
	ImageMatchInfos &operator=(const ImageMatchInfos &infos);
};

class MatchGraph {
	struct MatchGraphEdge {
		int objectImage;
		int sceneImage;
		double confidence;
	};

	inline static bool compareMatchGraphEdge(const MatchGraphEdge &first, const MatchGraphEdge &second)
	{
		return first.confidence > second.confidence;
	}

public:
	MatchGraph(const ImagesRegistry &images);
	void createScenes(std::vector<Scene> &scenes);

private:
	bool matchImages(const ImageDescriptor &sceneDescriptor, const ImageDescriptor &objectDescriptor);
	void computeHomography(const ImageDescriptor &sceneDescriptor, const ImageDescriptor &objectDescriptor);
	void findConnexComponents(std::vector<std::vector<bool>> &connexComponents);
	void findSpanningTree(std::list<MatchGraphEdge> &matchGraphEdges, std::vector<std::vector<bool>> &matchSpanningTreeEdges);
	void markNodeDepth(std::vector<int> &nodeDepth, std::vector<std::vector<bool>> &matchSpanningTreeEdges);
	void makeFinalSceneTree(int treeCenter, std::vector<std::vector<bool>> &matchSpanningTreeEdges, Scene &scene);

	std::vector<std::vector<ImageMatchInfos>> _matchInfos;
	std::list<MatchGraphEdge> _matchGraphEdges;
};

#endif
