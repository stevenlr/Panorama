#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <list>
#include <bitset>
#include <set>
#include <queue>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Scene.h"
#include "ImageMatching.h"
#include "Calibration.h"

#define NB_IMAGES 6

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
	initModule_features2d();
	initModule_nonfree();

	assert(NB_IMAGES > 1);

	Scene scene(NB_IMAGES);

	string sourceImagesNames[NB_IMAGES];

	if (false) {
		sourceImagesNames[0] = "balcony0";
		sourceImagesNames[1] = "balcony1";
		sourceImagesNames[2] = "balcony2";
	} else if (false) {
		sourceImagesNames[0] = "office1";
		sourceImagesNames[1] = "office2";
		sourceImagesNames[2] = "office4";
		sourceImagesNames[3] = "office3";
	} else {
		sourceImagesNames[0] = "building1";
		sourceImagesNames[1] = "building2";
		sourceImagesNames[2] = "building3";
		sourceImagesNames[3] = "building4";
		sourceImagesNames[4] = "building5";
		sourceImagesNames[5] = "building6";
	}

	for (int i = 0; i < NB_IMAGES; ++i) {
		Mat img = imread("../source_images/" + sourceImagesNames[i] + ".jpg");

		if (!img.data) {
			cerr << "Error when opening image " << sourceImagesNames[i] << endl;
			return 1;
		}

		scene.setImage(i, img);
	}

	Ptr<FeatureDetector> featureDetector = FeatureDetector::create("SIFT");
	Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create("SIFT");

	ImageDescriptor descriptors[NB_IMAGES];

	for (int i = 0; i < NB_IMAGES; ++i) {
		descriptors[i].image = i;
		descriptors[i].width = scene.getImage(i).size().width;
		descriptors[i].height = scene.getImage(i).size().height;
		featureDetector->detect(scene.getImageBW(i), descriptors[i].keypoints);
		descriptorExtractor->compute(scene.getImageBW(i), descriptors[i].keypoints, descriptors[i].featureDescriptor);
	}

	list<MatchGraphEdge> matchGraphEdges;
	map<pair<int, int>, ImageMatchInfos> matchInfosMap;
	vector<double> focalLengths;

	for (int i = 0; i < NB_IMAGES; ++i) {
		for (int j = i + 1; j < NB_IMAGES; ++j) {
			if (i == j) {
				continue;
			}

			ImageMatchInfos matchInfos = matchImages(descriptors[j], descriptors[i]);

			if (matchInfos.matches.size() >= 4) {
				computeHomography(descriptors[j], descriptors[i], matchInfos);

				if (matchInfos.confidence > 1) {
					ImageMatchInfos matchInfos2 = matchInfos;

					matchInfos2.homography = matchInfos2.homography.inv();

					findFocalLength(matchInfos.homography, focalLengths);
					findFocalLength(matchInfos2.homography, focalLengths);

					matchGraphEdges.push_back(make_pair(make_pair(i, j), matchInfos.confidence));
					matchGraphEdges.push_back(make_pair(make_pair(j, i), matchInfos2.confidence));

					matchInfosMap[make_pair(i, j)] = matchInfos;
					matchInfosMap[make_pair(j, i)] = matchInfos2;
				}
			}
		}
	}

	if (focalLengths.size() < 1) {
		cout << "Not enough images" << endl;
		cin.get();
		return 1;
	}
	
	int connexComponents[NB_IMAGES];
	bool fullyConnex = false;

	for (int i = 0; i < NB_IMAGES; ++i) {
		connexComponents[i] = i;
	}

	vector<vector<bool>> matchSpanningTreeEdges(NB_IMAGES);

	for (int i = 0; i < NB_IMAGES; ++i) {
		matchSpanningTreeEdges[i].resize(NB_IMAGES);

		for (int j = 0; j < NB_IMAGES; ++j) {
			matchSpanningTreeEdges[i][j] = false;
		}
	}

	matchGraphEdges.sort(compareMatchGraphEdge);

	while (!matchGraphEdges.empty() && !fullyConnex) {
		MatchGraphEdge elt(matchGraphEdges.front());
		int objectImage = elt.first.first;
		int sceneImage = elt.first.second;

		matchGraphEdges.pop_front();

		if (connexComponents[objectImage] == connexComponents[sceneImage]) {
			continue;
		}

		fullyConnex = true;
		matchSpanningTreeEdges[objectImage][sceneImage] = true;
		matchSpanningTreeEdges[sceneImage][objectImage] = true;

		for (int i = 0; i < NB_IMAGES; ++i) {
			if (connexComponents[i] == connexComponents[objectImage]) {
				connexComponents[i] = connexComponents[sceneImage];
			}

			if (connexComponents[i] != connexComponents[0]) {
				fullyConnex = false;
			}
		}
	}

	int nodeDepth[NB_IMAGES];

	for (int i = 0; i < NB_IMAGES; ++i) {
		nodeDepth[i] = numeric_limits<int>::max();
	}

	for (int i = 0; i < NB_IMAGES; ++i) {
		set<int> visited;
		queue<pair<int, int>> toVisit;
		int nbConnections = 0;

		for (int j = 0; j < NB_IMAGES; ++j) {
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

			for (int j = 0; j < NB_IMAGES; ++j) {
				if (matchSpanningTreeEdges[current][j] && visited.find(j) == visited.end()) {
					toVisit.push(make_pair(j, depth + 1));
				}
			}

			toVisit.pop();
		}
	}

	int treeCenter = max_element(nodeDepth, nodeDepth + NB_IMAGES) - nodeDepth;

	list<MatchGraphEdge> matchSpanningTree;

	{
		set<int> visited;
		queue<pair<int, int>> toVisit;

		toVisit.push(make_pair(treeCenter, -1));
		
		while (!toVisit.empty()) {
			int current = toVisit.front().first;
			int parent = toVisit.front().second;

			visited.insert(current);
			scene.setParent(current, parent);
			
			if (parent != -1) {
				scene.setTransform(current, matchInfosMap[make_pair(current, parent)].homography);
			} else {
				scene.setTransform(current, Mat::eye(Size(3, 3), CV_64F));
			}

			for (int j = 0; j < NB_IMAGES; ++j) {
				if (matchSpanningTreeEdges[current][j] && visited.find(j) == visited.end()) {
					toVisit.push(make_pair(j, current));
				}
			}

			toVisit.pop();
		}
	}

	int projSizeX = 1024;
	int projSizeY = 512;
	double focalLength = getMedianFocalLength(focalLengths);

	for (int i = 0; i < NB_IMAGES; ++i) {
		Mat img = scene.getImage(i);
		Mat map(Size(projSizeX, projSizeY), CV_32FC2, Scalar(-1, -1));
		const Mat &homography = scene.getFullTransform(i);
		double fovx = atan2(img.size().width, focalLength);
		double fovy = atan2(img.size().height, focalLength);
		
		Mat pose;
		double rx, ry, rz;

		Mat homography2;
		Mat intrinsic = Mat::eye(Size(3, 3), CV_64F);

		intrinsic.at<double>(0, 0) = focalLength;
		intrinsic.at<double>(1, 1) = focalLength * img.size().width / img.size().height;
		intrinsic.at<double>(0, 2) = -img.size().width / 2;
		intrinsic.at<double>(1, 2) = -img.size().height / 2;

		homography2 = intrinsic.inv() * homography * intrinsic;

		cameraPoseFromHomography(homography2, pose);
		findAnglesFromPose(pose, rx, ry, rz);

		cout << rx << " " << ry << " " << rz << endl;

		double roll = rz;

		for (int x = 0; x < projSizeX; ++x) {
			double angleX = ((double) x / projSizeX - 0.5) * PI * 2 - ry;

			if (angleX < -fovx || angleX > fovx) {
				continue;
			}

			for (int y = 0; y < projSizeY; ++y) {
				double angleY = (((double) y / projSizeY - 0.5) * PI - rx) * 0.99;

				if (angleY < -fovy || angleY > fovy) {
					continue;
				}

				double projX = (tan(angleX) * cos(roll) + tan(angleY) * sin(roll) / cos(angleX)) * focalLength;
				double projY = (tan(angleY) * cos(roll) / cos(angleX) - sin(roll) * tan(angleX)) * focalLength;

				map.at<Vec2f>(y, x)[0] = static_cast<float>(projX + img.size().width / 2);
				map.at<Vec2f>(y, x)[1] = static_cast<float>(projY + img.size().height / 2);
			}
		}

		Mat img2(Size(projSizeX, projSizeY), CV_8UC3, Scalar(0, 0, 0));

		remap(img, img2, map, Mat(), INTER_LINEAR, BORDER_TRANSPARENT);
		//namedWindow(sourceImagesNames[i], WINDOW_AUTOSIZE);
		//imshow(sourceImagesNames[i], img2);
	}

	Mat panorama = scene.composePanorama();

	namedWindow("output", WINDOW_AUTOSIZE);
	imshow("output", panorama);
	imwrite("output.jpg", panorama);

	waitKey(0);

	return 0;
}
