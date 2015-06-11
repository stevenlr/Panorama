#include "MatchGraph.h"

#include <algorithm>
#include <set>
#include <queue>
#include <iostream>
#include <thread>
#include <ctime>

#include <opencv2/calib3d/calib3d.hpp>

#include "Calibration.h"
#include "Scene.h"
#include "ImageSequence.h"

using namespace std;
using namespace cv;

#define CONFIDENCE_THRESHOLD 1.0

ImageMatchInfos::ImageMatchInfos()
{
	confidence = 0;
	nbInliers = 0;
	nbOverlaps = 0;

	homography = Mat();
}

ImageMatchInfos::ImageMatchInfos(const ImageMatchInfos &infos)
{
	confidence = infos.confidence;
	nbInliers = infos.nbInliers;
	nbOverlaps = infos.nbOverlaps;

	homography = infos.homography.clone();

	matches.resize(infos.matches.size());
	copy(infos.matches.begin(), infos.matches.end(), matches.begin());

	inliersMask.resize(infos.inliersMask.size());
	copy(infos.inliersMask.begin(), infos.inliersMask.end(), inliersMask.begin());
}

ImageMatchInfos &ImageMatchInfos::operator=(const ImageMatchInfos &infos)
{
	if (this == &infos) {
		return *this;
	}

	confidence = infos.confidence;
	nbInliers = infos.nbInliers;
	nbOverlaps = infos.nbOverlaps;

	homography = infos.homography.clone();

	matches.resize(infos.matches.size());
	copy(infos.matches.begin(), infos.matches.end(), matches.begin());

	inliersMask.resize(infos.inliersMask.size());
	copy(infos.inliersMask.begin(), infos.inliersMask.end(), inliersMask.begin());

	return *this;
}

void MatchGraph::pairwiseMatch(queue<PairwiseMatchTask> &tasks)
{
	while (!tasks.empty()) {
		PairwiseMatchTask task = tasks.front();

		const ImageDescriptor &sceneDescriptor = *(task.first);
		const ImageDescriptor &objectDescriptor = *(task.second);

		tasks.pop();

		_printMutex.lock();
		_progress++;
		cout << "\rPairwise matching " << static_cast<int>((static_cast<float>(_progress) / _totalTasks * 100)) << "%" << flush;
		_printMutex.unlock();

		if (matchImages(sceneDescriptor, objectDescriptor)) {
			computeHomography(sceneDescriptor, objectDescriptor);
			_matchInfosMutex.lock();

			ImageMatchInfos *matchInfos1 = &_matchInfos[sceneDescriptor.image][objectDescriptor.image];

			if (matchInfos1->confidence > CONFIDENCE_THRESHOLD) {
				MatchGraphEdge edge1;
				/*MatchGraphEdge edge2;

				ImageMatchInfos *matchInfos2 = &_matchInfos[objectDescriptor.image][sceneDescriptor.image];

				new(matchInfos2) ImageMatchInfos(*matchInfos1);*/

				edge1.sceneImage = sceneDescriptor.image;
				edge1.objectImage = objectDescriptor.image;
				edge1.confidence = matchInfos1->confidence;

				/*edge2.sceneImage = objectDescriptor.image;
				edge2.objectImage = sceneDescriptor.image;
				edge2.confidence = matchInfos1->confidence;
				matchInfos2->homography = matchInfos2->homography.inv();*/

				_matchGraphEdges.push_back(edge1);
				//_matchGraphEdges.push_back(edge2);
			}

			_matchInfosMutex.unlock();
		}
	}
}

MatchGraph::MatchGraph(const ImagesRegistry &images)
{
	int nbImages = images.getNbImages();
	int nbThreads = thread::hardware_concurrency();
	int threadId = 0;
	vector<thread> threads(nbThreads);
	vector<queue<PairwiseMatchTask>> taskQueues(nbThreads);

	_totalTasks = nbImages * (nbImages - 1);
	_progress = 0;
	_matchInfos.resize(nbImages);

	cout << nbThreads << " threads" << endl;

	for (int i = 0; i < nbImages; ++i) {
		_matchInfos[i].resize(nbImages);
	}

	for (int i = 0; i < nbImages; ++i) {
		for (int j = 0; j < nbImages; ++j) {
			if (i == j) {
				continue;
			}

			taskQueues[threadId].push(make_pair(&images.getDescriptor(i), &images.getDescriptor(j)));
			threadId = (threadId + 1) % nbThreads;
		}
	}

	clock_t start = clock();

	for (int i = 0; i < nbThreads; ++i) {
		new(&threads[i]) thread(&MatchGraph::pairwiseMatch, std::ref(*this), taskQueues[i]);
	}

	for (int i = 0; i < nbThreads; ++i) {
		threads[i].join();
	}

	float elapsedTime = static_cast<float>(clock() - start) / CLOCKS_PER_SEC;

	cout << endl << "Pairwise matching average: " << (elapsedTime / _totalTasks * nbThreads) << "s" << endl;
	cout << "Pairwise matching total (multi-threaded): " << elapsedTime << "s" << endl;
}

bool MatchGraph::matchImages(const ImageDescriptor &sceneDescriptor, const ImageDescriptor &objectDescriptor)
{
	const double confidence = 0.5;
	ImageMatchInfos &matchInfos = _matchInfos[sceneDescriptor.image][objectDescriptor.image];
	Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create("BruteForce");
	vector<vector<DMatch>> matches;
	set<pair<int, int>> matchesSet;

	descriptorMatcher->knnMatch(objectDescriptor.featureDescriptor, sceneDescriptor.featureDescriptor, matches, 2);
	descriptorMatcher->clear();

	for (size_t i = 0; i < matches.size(); ++i) {
		if (matches[i].size() < 2) {
			continue;
		}

		const DMatch &m0 = matches[i][0];
		const DMatch &m1 = matches[i][1];

		if (m0.distance < (1 - confidence) * m1.distance) {
			_matchInfosMutex.lock();
			matchesSet.insert(make_pair(m0.trainIdx, m0.queryIdx));
			matchInfos.matches.push_back(make_pair(m0.trainIdx, m0.queryIdx));
			_matchInfosMutex.unlock();
		}
	}

	/*matches.clear();
	descriptorMatcher->knnMatch(sceneDescriptor.featureDescriptor, objectDescriptor.featureDescriptor, matches, 2);
	descriptorMatcher->clear();

	for (size_t i = 0; i < matches.size(); ++i) {
		if (matches[i].size() < 2) {
			continue;
		}

		const DMatch &m0 = matches[i][0];
		const DMatch &m1 = matches[i][1];

		if (m0.distance < (1 - confidence) * m1.distance) {
			if (matchesSet.find(make_pair(m0.queryIdx, m0.trainIdx)) == matchesSet.end()) {
				_matchInfosMutex.lock();
				matchInfos.matches.push_back(make_pair(m0.queryIdx, m0.trainIdx));
				_matchInfosMutex.unlock();
			}
		}
	}*/

	_matchInfosMutex.lock();

	if (matchInfos.matches.size() < 4) {
		_matchInfosMutex.unlock();
		return false;
	}

	_matchInfosMutex.unlock();
	return true;
}

void MatchGraph::computeHomography(const ImageDescriptor &sceneDescriptor, const ImageDescriptor &objectDescriptor)
{
	ImageMatchInfos &match = _matchInfos[sceneDescriptor.image][objectDescriptor.image];
	vector<Point2f> points[2];
	vector<pair<int, int>>::const_iterator matchesIt = match.matches.cbegin();

	match.nbInliers = 0;
	match.nbOverlaps = 0;
	match.inliersMask.clear();

	while (matchesIt != match.matches.cend()) {
		const pair<int, int> &m = *matchesIt++;

		Point2f scenePoint = sceneDescriptor.keypoints[m.first].pt;
		Point2f objectPoint = objectDescriptor.keypoints[m.second].pt;

		scenePoint.x -= sceneDescriptor.width / 2;
		scenePoint.y -= sceneDescriptor.height / 2;

		objectPoint.x -= objectDescriptor.width / 2;
		objectPoint.y -= objectDescriptor.height / 2;

		points[0].push_back(scenePoint);
		points[1].push_back(objectPoint);
	}

	vector<uchar> inliersMask;

	Mat homography = findHomography(points[1], points[0], CV_RANSAC, 3.0, inliersMask);

	vector<uchar>::iterator inliersMaskIt;
	vector<Point2f>::const_iterator pointsIt[2];

	inliersMaskIt = inliersMask.begin();
	pointsIt[0] = points[0].begin();
	pointsIt[1] = points[1].begin();

	while (inliersMaskIt != inliersMask.end()) {
		if (*inliersMaskIt++) {
			pointsIt[0]++;
			pointsIt[1]++;
		} else {
			pointsIt[0] = points[0].erase(pointsIt[0]);
			pointsIt[1] = points[1].erase(pointsIt[1]);
		}
	}

	vector<uchar> inliersMask2;

	homography = findHomography(points[1], points[0], CV_RANSAC, 3.0, inliersMask2);

	matchesIt = match.matches.cbegin();
	inliersMaskIt = inliersMask.begin();

	int nbOverlaps = 0;
	int nbInliers = 0;
	vector<uchar>::const_iterator inliersMask2It = inliersMask2.cbegin();

	while (matchesIt != match.matches.cend()) {
		if (*inliersMaskIt) {
			Point2d point = objectDescriptor.keypoints[matchesIt->second].pt;
			Mat pointH = Mat::ones(Size(1, 3), CV_64F);

			pointH.at<double>(0, 0) = point.x - objectDescriptor.width / 2;
			pointH.at<double>(1, 0) = point.y - objectDescriptor.height / 2;

			pointH = homography * pointH;
			point.x = pointH.at<double>(0, 0) / pointH.at<double>(2, 0) + objectDescriptor.width / 2;
			point.y = pointH.at<double>(1, 0) / pointH.at<double>(2, 0) + objectDescriptor.height / 2;

			if (point.x >= 0 && point.y >= 0 && point.x < sceneDescriptor.width && point.y < sceneDescriptor.height) {
				nbOverlaps++;

				if (*inliersMask2It) {
					nbInliers++;
				} else {
					*inliersMaskIt = 0;
				}
			}

			inliersMask2It++;
		}

		matchesIt++;
		inliersMaskIt++;
	}

	_matchInfosMutex.lock();
	match.inliersMask = inliersMask;
	match.nbInliers = nbInliers;
	match.nbOverlaps = nbOverlaps;
	match.homography = homography;
	match.confidence = match.nbInliers / (8.0 + 0.3 * match.nbOverlaps);
	_matchInfosMutex.unlock();
}

void MatchGraph::findConnexComponents(vector<vector<bool>> &connexComponents)
{
	int nbImages = _matchInfos.size();
	vector<int> connexComponentsIds(nbImages);

	for (int i = 0; i < nbImages; ++i) {
		connexComponentsIds[i] = i;
	}

	_matchGraphEdges.sort(compareMatchGraphEdge);

	for (const MatchGraphEdge &edge : _matchGraphEdges) {
		if (edge.confidence < CONFIDENCE_THRESHOLD) {
			continue;
		}

		if (connexComponentsIds[edge.objectImage] == connexComponentsIds[edge.sceneImage]) {
			continue;
		}

		for (int i = 0; i < nbImages; ++i) {
			if (connexComponentsIds[i] == connexComponentsIds[edge.objectImage]) {
				connexComponentsIds[i] = connexComponentsIds[edge.sceneImage];
			}
		}
	}

	set<int> componentsRegistered;

	for (int i = 0; i < nbImages; ++i) {
		int componentId = connexComponentsIds[i];

		if (componentsRegistered.find(componentId) != componentsRegistered.end()) {
			continue;
		}

		vector<bool> nodes(nbImages);

		nodes[i] = true;

		for (int j = i + 1; j < nbImages; ++j) {
			if (connexComponentsIds[j] == componentId) {
				nodes[j] = true;
			}
		}

		connexComponents.push_back(nodes);
		componentsRegistered.insert(componentId);
	}
}

void MatchGraph::findSpanningTree(list<MatchGraphEdge> &matchGraphEdges, vector<vector<bool>> &matchSpanningTreeEdges)
{
	int nbImages = _matchInfos.size();
	vector<int> connexComponents(nbImages);

	matchGraphEdges.sort(compareMatchGraphEdge);
	matchSpanningTreeEdges.resize(nbImages);

	for (int i = 0; i < nbImages; ++i) {
		connexComponents[i] = i;
		matchSpanningTreeEdges[i].resize(nbImages);

		for (int j = 0; j < nbImages; ++j) {
			matchSpanningTreeEdges[i][j] = false;
		}
	}

	while (!matchGraphEdges.empty()) {
		MatchGraphEdge edge(matchGraphEdges.front());

		matchGraphEdges.pop_front();

		if (connexComponents[edge.objectImage] == connexComponents[edge.sceneImage]) {
			continue;
		}

		matchSpanningTreeEdges[edge.objectImage][edge.sceneImage] = true;
		matchSpanningTreeEdges[edge.sceneImage][edge.objectImage] = true;

		for (int i = 0; i < nbImages; ++i) {
			if (connexComponents[i] == connexComponents[edge.objectImage]) {
				connexComponents[i] = connexComponents[edge.sceneImage];
			}
		}
	}
}

void MatchGraph::markNodeDepth(vector<int> &nodeDepth, vector<vector<bool>> &matchSpanningTreeEdges)
{
	int nbImages = _matchInfos.size();

	nodeDepth.resize(nbImages);

	for (int i = 0; i < nbImages; ++i) {
		nodeDepth[i] = -1;

		for (int j = 0; j < nbImages; ++j) {
			if (matchSpanningTreeEdges[i][j] || matchSpanningTreeEdges[j][i]) {
				nodeDepth[i] = numeric_limits<int>::max();
				break;
			}
		}
	}

	for (int i = 0; i < nbImages; ++i) {
		set<int> visited;
		queue<pair<int, int>> toVisit;
		int nbConnections = 0;

		for (int j = 0; j < nbImages; ++j) {
			if (matchSpanningTreeEdges[i][j]) {
				nbConnections++;
			}
		}

		if (nbConnections != 1) {
			continue;
		}

		toVisit.push(make_pair(i, 0));
		
		while (!toVisit.empty()) {
			int current = toVisit.front().first;
			int depth = toVisit.front().second;

			nodeDepth[current] = min(nodeDepth[current], depth);
			visited.insert(current);

			for (int j = 0; j < nbImages; ++j) {
				if (matchSpanningTreeEdges[current][j] && visited.find(j) == visited.end()) {
					toVisit.push(make_pair(j, depth + 1));
				}
			}

			toVisit.pop();
		}
	}
}

void MatchGraph::makeFinalSceneTree(int treeCenter, vector<vector<bool>> &matchSpanningTreeEdges, Scene &scene, ImageSequence &sequence)
{
	set<int> visited;
	queue<pair<int, int>> toVisit;
	int nbImages = _matchInfos.size();

	toVisit.push(make_pair(treeCenter, -1));
		
	while (!toVisit.empty()) {
		int current = toVisit.front().first;
		int parent = toVisit.front().second;
		int img = scene.getIdInScene(sequence.getKeyFrame(current));

		visited.insert(current);
			
		if (parent != -1) {
			scene.setParent(img, scene.getIdInScene(sequence.getKeyFrame(parent)));
			scene.setTransform(img, _matchInfos[parent][current].homography);
		} else {
			scene.setParent(img, -1);
			scene.setTransform(img, Mat::eye(Size(3, 3), CV_64F));
		}

		for (int j = 0; j < nbImages; ++j) {
			if (matchSpanningTreeEdges[current][j] && visited.find(j) == visited.end()) {
				toVisit.push(make_pair(j, current));
			}
		}

		toVisit.pop();
	}
}

void MatchGraph::createScenes(std::vector<Scene> &scenes, ImageSequence &sequence)
{
	vector<vector<bool>> connexComponents;

	findConnexComponents(connexComponents);

	int nbComponents = connexComponents.size();
	int nbImages = _matchInfos.size();

	scenes.resize(nbComponents);

	for (int i = 0; i < nbComponents; ++i) {
		list<MatchGraphEdge> edges;
		int imageId = 0;
		Scene &scene = scenes[i];
		vector<double> focalLengths;

		for (int j = 0; j < nbImages; ++j) {
			if (connexComponents[i][j]) {
				scene.addImage(sequence.getKeyFrame(j));
			}
		}

		for (const MatchGraphEdge &edge : _matchGraphEdges) {
			if (connexComponents[i][edge.objectImage] && connexComponents[i][edge.sceneImage] && edge.confidence > CONFIDENCE_THRESHOLD) {
				edges.push_back(edge);
				findFocalLength(_matchInfos[edge.sceneImage][edge.objectImage].homography, focalLengths);
			}
		}

		if (focalLengths.size() >= 1) {
			scene.setEstimatedFocalLength(getMedianFocalLength(focalLengths));
		}

		vector<vector<bool>> spanningTreeEdges;
		vector<int> nodeDepth;

		findSpanningTree(edges, spanningTreeEdges);
		markNodeDepth(nodeDepth, spanningTreeEdges);

		int treeCenter = max_element(nodeDepth.begin(), nodeDepth.end()) - nodeDepth.begin();

		if (nodeDepth[treeCenter] == -1) {
			int img = sequence.getKeyFrame(treeCenter);
			scene.setParent(img, -1);
			scene.setTransform(img, Mat::eye(Size(3, 3), CV_64F));
			continue;
		}

		makeFinalSceneTree(treeCenter, spanningTreeEdges, scene, sequence);
	}
}

const ImageMatchInfos &MatchGraph::getImageMatchInfos(int sceneImage, int objectImage) const
{
	int nbImages = _matchInfos.size();

	assert(sceneImage >= 0 && sceneImage < nbImages);
	assert(objectImage >= 0 && objectImage < nbImages);

	return _matchInfos[sceneImage][objectImage];
}
