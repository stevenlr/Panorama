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

#define DATASET 1

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
	initModule_features2d();
	initModule_nonfree();

	vector<string> sourceImagesNames;

#if DATASET == 1
	sourceImagesNames.push_back("balcony0");
	sourceImagesNames.push_back("balcony1");
	sourceImagesNames.push_back("balcony2");
#elif DATASET == 2
	sourceImagesNames.push_back("office1");
	sourceImagesNames.push_back("office2");
	sourceImagesNames.push_back("office4");
	sourceImagesNames.push_back("office3");
#elif DATASET == 3
	sourceImagesNames.push_back("building1");
	sourceImagesNames.push_back("building2");
	sourceImagesNames.push_back("building3");
	sourceImagesNames.push_back("building4");
	sourceImagesNames.push_back("building5");
	sourceImagesNames.push_back("building6");
#elif DATASET == 4
	sourceImagesNames.push_back("mountain1");
	sourceImagesNames.push_back("mountain2");
#elif DATASET == 5
	sourceImagesNames.push_back("bus1");
	sourceImagesNames.push_back("bus2");
#elif DATASET == 6
	sourceImagesNames.push_back("cliff1");
	sourceImagesNames.push_back("cliff2");
	sourceImagesNames.push_back("cliff3");
	sourceImagesNames.push_back("cliff4");
	sourceImagesNames.push_back("cliff5");
	sourceImagesNames.push_back("cliff6");
	sourceImagesNames.push_back("cliff7");
#else
	return 1;
#endif

	int nbImages = sourceImagesNames.size();
	Scene scene(nbImages);

	for (int i = 0; i < nbImages; ++i) {
		Mat img = imread("../source_images/" + sourceImagesNames[i] + ".jpg");

		if (!img.data) {
			cerr << "Error when opening image " << sourceImagesNames[i] << endl;
			return 1;
		}

		scene.setImage(i, img);
	}

	Ptr<FeatureDetector> featureDetector = FeatureDetector::create("SIFT");
	Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create("SIFT");

	vector<ImageDescriptor> descriptors(nbImages);

	for (int i = 0; i < nbImages; ++i) {
		descriptors[i].image = i;
		descriptors[i].width = scene.getImage(i).size().width;
		descriptors[i].height = scene.getImage(i).size().height;
		featureDetector->detect(scene.getImageBW(i), descriptors[i].keypoints);
		descriptorExtractor->compute(scene.getImageBW(i), descriptors[i].keypoints, descriptors[i].featureDescriptor);
	}

	list<MatchGraphEdge> matchGraphEdges;
	map<pair<int, int>, ImageMatchInfos> matchInfosMap;
	vector<double> focalLengths;

	for (int i = 0; i < nbImages; ++i) {
		for (int j = i + 1; j < nbImages; ++j) {
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
	
	vector<int> connexComponents(nbImages);
	bool fullyConnex = false;

	for (int i = 0; i < nbImages; ++i) {
		connexComponents[i] = i;
	}

	vector<vector<bool>> matchSpanningTreeEdges(nbImages);

	for (int i = 0; i < nbImages; ++i) {
		matchSpanningTreeEdges[i].resize(nbImages);

		for (int j = 0; j < nbImages; ++j) {
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

		for (int i = 0; i < nbImages; ++i) {
			if (connexComponents[i] == connexComponents[objectImage]) {
				connexComponents[i] = connexComponents[sceneImage];
			}

			if (connexComponents[i] != connexComponents[0]) {
				fullyConnex = false;
			}
		}
	}

	vector<int> nodeDepth(nbImages);

	for (int i = 0; i < nbImages; ++i) {
		nodeDepth[i] = numeric_limits<int>::max();
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

	int treeCenter = max_element(nodeDepth.begin(), nodeDepth.end()) - nodeDepth.begin();

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

			for (int j = 0; j < nbImages; ++j) {
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
	Mat finalImage(Size(projSizeX, projSizeY), scene.getImage(0).type());

	cout << focalLength << endl << endl;

	for (int i = 0; i < nbImages; ++i) {
		Mat img = scene.getImage(i);
		Mat map(Size(projSizeX, projSizeY), CV_32FC2, Scalar(-1, -1));
		const Mat &homography = scene.getFullTransform(i);
		double fovx = 2 * atan2(img.size().width, focalLength);
		double fovy = 2 * atan2(img.size().height, focalLength);

		cout << fovx * 180 / PI << " " << fovy * 180 / PI << endl;
		
		Mat pose;
		double rx, ry, rz;

		Mat homography2;
		Mat intrinsic = Mat::eye(Size(3, 3), CV_64F);

		intrinsic.at<double>(0, 0) = focalLength;
		intrinsic.at<double>(1, 1) = focalLength * img.size().width / img.size().height;
		intrinsic.at<double>(0, 2) = -img.size().width / 2;
		intrinsic.at<double>(1, 2) = -img.size().height / 2;

		homography2 = intrinsic.inv() * homography * intrinsic;
		findAnglesFromPose(homography2, rx, ry, rz);

		cout << rx << " " << ry << " " << rz << endl << endl;

		double roll = rx;
		double theta = -ry;
		double phi = -rz;

		for (int x = 0; x < projSizeX; ++x) {
			double angleX = ((double) x / projSizeX - 0.5) * PI * 2 + theta;

			if (angleX < -fovx || angleX > fovx) {
				continue;
			}

			for (int y = 0; y < projSizeY; ++y) {
				double angleY = (((double) y / projSizeY - 0.5) * PI + phi * 4) * 0.99;

				if (angleY < -fovy || angleY > fovy) {
					continue;
				}

				double projX = (tan(angleX) * cos(roll) + tan(angleY) * sin(roll) / cos(angleX)) * focalLength / 2;
				double projY = (tan(angleY) * cos(roll) / cos(angleX) - sin(roll) * tan(angleX)) * focalLength / 2;

				map.at<Vec2f>(y, x)[0] = static_cast<float>(projX + img.size().width / 2);
				map.at<Vec2f>(y, x)[1] = static_cast<float>(projY + img.size().height / 2);
			}
		}

		Mat maskNormal = Mat::ones(img.size(), CV_8U);
		Mat maskSpherical(Size(projSizeX, projSizeY), CV_8U, Scalar(0));
		Mat img2(Size(projSizeX, projSizeY), CV_8UC3, Scalar(0, 0, 0));

		remap(maskNormal, maskSpherical, map, Mat(), INTER_LINEAR, BORDER_TRANSPARENT);
		remap(img, img2, map, Mat(), INTER_LINEAR, BORDER_TRANSPARENT);
		img2.copyTo(finalImage, maskSpherical);
	}

	namedWindow("output spherical", WINDOW_AUTOSIZE);
	imshow("output spherical", finalImage);

	/*Mat panorama = scene.composePanorama();

	namedWindow("output", WINDOW_AUTOSIZE);
	imshow("output", panorama);
	imwrite("output.jpg", panorama);*/

	waitKey(0);

	return 0;
}
