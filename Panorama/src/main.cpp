#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <list>
#include <set>

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

	vector<ImageDescriptor> descriptors(nbImages);

	extractFeatures(scene, descriptors);

	list<MatchGraphEdge> matchGraphEdges;
	map<pair<int, int>, ImageMatchInfos> matchInfosMap;
	vector<double> focalLengths;

	pairwiseMatch(scene, descriptors, matchGraphEdges, matchInfosMap, focalLengths);

	if (focalLengths.size() < 1) {
		cout << "Not enough images" << endl;
		cin.get();
		return 1;
	}

	scene.makeSceneGraph(matchGraphEdges, matchInfosMap);

	int projSizeX = 1024;
	int projSizeY = 512;
	double focalLength = getMedianFocalLength(focalLengths);
	Mat finalImage(Size(projSizeX, projSizeY), scene.getImage(0).type());

	for (int i = 0; i < nbImages; ++i) {
		Mat img = scene.getImage(i);
		Mat map(Size(projSizeX, projSizeY), CV_32FC2, Scalar(-1, -1));
		Mat homography = scene.getFullTransform(i).clone();

		Mat translation = Mat::eye(Size(3, 3), CV_64F);

		translation.at<double>(0, 2) = -img.size().width / 2;
		translation.at<double>(1, 2) = -img.size().height / 2;

		homography = homography * translation;

		Mat invHomography = homography.inv();

		for (int x = 0; x < projSizeX; ++x) {
			double angleX = ((double) x / projSizeX - 0.5) * PI;

			for (int y = 0; y < projSizeY; ++y) {
				double angleY = ((double) y / projSizeY - 0.5) * PI / 2;

				Mat spacePoint = Mat::zeros(Size(1, 3), CV_64F);
				spacePoint.at<double>(0, 0) = sin(angleX) * cos(angleY) * focalLength;
				spacePoint.at<double>(1, 0) = sin(angleY) * focalLength;
				spacePoint.at<double>(2, 0) = cos(angleX) * cos(angleY);

				Mat transformedPoint = invHomography * spacePoint;
				double projX = transformedPoint.at<double>(0, 0) / transformedPoint.at<double>(2, 0);
				double projY = transformedPoint.at<double>(1, 0) / transformedPoint.at<double>(2, 0);

				map.at<Vec2f>(y, x)[0] = static_cast<float>(projX);
				map.at<Vec2f>(y, x)[1] = static_cast<float>(projY);
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
	imwrite("output.jpg", finalImage);

	//Mat panorama = scene.composePanorama();

	/*namedWindow("output", WINDOW_AUTOSIZE);
	imshow("output", panorama);
	imwrite("output.jpg", panorama);*/

	waitKey(0);

	return 0;
}
