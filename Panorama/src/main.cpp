#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Scene.h"

#define NB_IMAGES 2

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
	initModule_features2d();
	initModule_nonfree();

	Scene scene(NB_IMAGES);

	string sourceImagesNames[NB_IMAGES];
	sourceImagesNames[0] = "mountain1";
	sourceImagesNames[1] = "mountain2";

	for (int i = 0; i < NB_IMAGES; ++i) {
		Mat img = imread("../source_images/" + sourceImagesNames[i] + ".jpg");

		if (!img.data) {
			cout << "Error when opening image " << sourceImagesNames[i] << endl;
			return 1;
		}

		scene.setImage(i, img);
	}

	scene.setParent(0, -1);
	scene.setTransform(0, Mat::eye(3, 3, CV_64F));

	Ptr<FeatureDetector> featureDetector = FeatureDetector::create("SIFT");
	Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create("SIFT");

	vector<KeyPoint> keypoints[NB_IMAGES];
	Mat featureDescriptors[NB_IMAGES];

	for (int i = 0; i < NB_IMAGES; ++i) {
		featureDetector->detect(scene.getImage(i), keypoints[i]);
		descriptorExtractor->compute(scene.getImage(i), keypoints[i], featureDescriptors[i]);
	}

	const int sceneImage = 0;
	const int objectImage = 1;

	Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create("FlannBased");
	vector<DMatch> matches;

	descriptorMatcher->match(featureDescriptors[objectImage], featureDescriptors[sceneImage], matches);

	float minMatchDist = 10000;

	for (int i = 0; i < matches.size(); ++i) {
		float dist = matches[i].distance;

		if (dist < minMatchDist) {
			minMatchDist = dist;
		}
	}

	vector<DMatch> goodMatches;

	for (int i = 0; i < matches.size(); ++i) {
		if (matches[i].distance < 2.5 * minMatchDist) {
			goodMatches.push_back(matches[i]);
		}
	}

	vector<Point2f> points[2];

	for (int i = 0; i < goodMatches.size(); ++i) {
		points[0].push_back(keypoints[sceneImage][goodMatches[i].trainIdx].pt);
		points[1].push_back(keypoints[objectImage][goodMatches[i].queryIdx].pt);
	}

	Mat homography = findHomography(points[1], points[0], CV_RANSAC);

	scene.setParent(objectImage, sceneImage);
	scene.setTransform(objectImage, homography);

	Mat panorama = scene.composePanorama();

	namedWindow("panorama", WINDOW_AUTOSIZE);
	imshow("panorama", panorama);

	waitKey(0);

	return 0;
}
