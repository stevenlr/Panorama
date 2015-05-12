#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <list>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Scene.h"
#include "ImageMatching.h"
#include "Calibration.h"

#define NB_IMAGES 3

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
	initModule_features2d();
	initModule_nonfree();

	assert(NB_IMAGES > 1);

	Scene scene(NB_IMAGES);

	string sourceImagesNames[NB_IMAGES];
	sourceImagesNames[0] = "office1";
	sourceImagesNames[1] = "office2";
	sourceImagesNames[2] = "office3";

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
		featureDetector->detect(scene.getImageBW(i), descriptors[i].keypoints);
		descriptorExtractor->compute(scene.getImageBW(i), descriptors[i].keypoints, descriptors[i].featureDescriptor);
	}

	list<MatchMatrixElement> matchMatrix;

	for (int i = 0; i < NB_IMAGES; ++i) {
		for (int j = 0; j < NB_IMAGES; ++j) {
			if (i == j) {
				continue;
			}

			ImageMatchInfos matchInfos = matchImages(descriptors[j], descriptors[i]);

			if (matchInfos.matches.size() >= 4) {
				matchMatrix.push_back(make_pair(make_pair(i, j), matchInfos));
				matchMatrix.push_back(make_pair(make_pair(j, i), matchInfos));
			}
		}
	}

	int imageMatched = 0;
	vector<double> focalLengths;

	matchMatrix.sort(compareMatchMatrixElements);

	while (imageMatched < NB_IMAGES - 1) {
		const MatchMatrixElement &elt = matchMatrix.front();
		int objectImage = elt.first.first;
		int sceneImage = elt.first.second;

		scene.setParent(objectImage, sceneImage);

		if (scene.checkCycle(objectImage)) {
			scene.setParent(objectImage, -1);
			matchMatrix.erase(matchMatrix.begin());
			continue;
		}

		Mat homography = computeHomography(descriptors[sceneImage], descriptors[objectImage], elt.second);
		Mat pose;
		double rx, ry, rz;

		cameraPoseFromHomography(homography, pose);
		findAnglesFromPose(pose, rx, ry, rz);
		findFocalLength(homography, focalLengths);

		scene.setTransform(objectImage, homography);
		
		list<MatchMatrixElement>::iterator it = matchMatrix.begin();

		while (it != matchMatrix.end()) {
			if (it->first.first == objectImage) {
				it = matchMatrix.erase(it);
			} else {
				++it;
			}
		}

		imageMatched++;
	}

	double focalLength = getMedianFocalLength(focalLengths);
	int projSizeX = 360 * 3;
	int projSizeY = 180 * 3;

	for (int i = 0; i < 1; ++i) {
		Mat img = scene.getImage(i);
		double hfovx = atan(img.size().width / focalLength / 2) * projSizeX / (PI * 2);
		double hfovy = atan(img.size().height / focalLength / 2) * projSizeY / (PI * 2);

		Mat map(Size(projSizeX, projSizeY), CV_32FC2, Scalar(-1));

		for (int x = 0; x < projSizeX; ++x) {
			double angleX = ((double) x / projSizeX - 0.5) * PI;
			double projX = tan(angleX) * focalLength;

			for (int y = 0; y < projSizeY; ++y) {
				double angleY = ((double) y / projSizeY - 0.5) * PI / 2;
				double projY = tan(angleY) * focalLength * img.size().width / img.size().height;

				Vec2f &elt = map.at<Vec2f>(x, y);

				map.at<Vec2f>(y, x)[0] = static_cast<float>(projX);
				map.at<Vec2f>(y, x)[1] = static_cast<float>(projY);
			}
		}

		Mat img2(Size(projSizeX, projSizeY), CV_8UC3);

		remap(img, img2, map, Mat(), INTER_LINEAR, BORDER_TRANSPARENT);
		namedWindow(sourceImagesNames[i], WINDOW_AUTOSIZE);
		imshow(sourceImagesNames[i], img2);
	}

	Mat panorama = scene.composePanorama();

	namedWindow("panorama", WINDOW_AUTOSIZE);
	imshow("panorama", panorama);

	imwrite("output.jpg", panorama);

	waitKey(0);

	return 0;
}
