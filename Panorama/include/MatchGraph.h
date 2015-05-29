#ifndef MATCH_GRAPH_H_
#define MATCH_GRAPH_H_

#include "ImagesRegistry.h"

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

	inline bool compareMatchGraphEdge(const MatchGraphEdge &first, const MatchGraphEdge &second)
	{
		return first.confidence > second.confidence;
	}

public:
	MatchGraph(const ImagesRegistry &images);

private:
	bool matchImages(const ImageDescriptor &sceneDescriptor, const ImageDescriptor &objectDescriptor);
	void computeHomography(const ImageDescriptor &sceneDescriptor, const ImageDescriptor &objectDescriptor);

	std::vector<std::vector<ImageMatchInfos>> _matchInfos;
};

#endif
