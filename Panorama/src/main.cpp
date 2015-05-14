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

	if (false) {
		sourceImagesNames[0] = "balcony0";
		sourceImagesNames[1] = "balcony1";
		sourceImagesNames[2] = "balcony2";
	} else {
		sourceImagesNames[0] = "office1";
		sourceImagesNames[1] = "office2";
		sourceImagesNames[2] = "office3";
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

	list<MatchMatrixElement> matchMatrix;
	vector<double> focalLengths;

	for (int i = 0; i < NB_IMAGES; ++i) {
		for (int j = 0; j < NB_IMAGES; ++j) {
			if (i == j) {
				continue;
			}

			ImageMatchInfos matchInfos = matchImages(descriptors[j], descriptors[i]);

			if (matchInfos.matches.size() >= 4) {
				computeHomography(descriptors[j], descriptors[i], matchInfos);
				matchMatrix.push_back(make_pair(make_pair(i, j), matchInfos));
				findFocalLength(matchInfos.homography, focalLengths);
			}
		}
	}

	if (focalLengths.size() < 1) {
		cout << "Not enough images" << endl;
		cin.get();
		return 1;
	}

	double focalLength = getMedianFocalLength(focalLengths);
	int imageMatched = 0;

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
		
		scene.setTransform(objectImage, elt.second.homography);
		
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

	int projSizeX = 1024;
	int projSizeY = 512;

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

		homography2 = intrinsic.inv() * homography.inv() * intrinsic;

		cameraPoseFromHomography(homography2, pose);
		findAnglesFromPose(pose, rx, ry, rz);

		double roll = rx;

		for (int x = 0; x < projSizeX; ++x) {
			double angleX = ((double) x / projSizeX - 0.5) * PI * 2 + ry;

			if (angleX < -fovx || angleX > fovx) {
				continue;
			}

			for (int y = 0; y < projSizeY; ++y) {
				double angleY = (((double) y / projSizeY - 0.5) * PI - rz) * 0.99;

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
		namedWindow(sourceImagesNames[i], WINDOW_AUTOSIZE);
		imshow(sourceImagesNames[i], img2);
	}

	Mat panorama = scene.composePanorama();

	imwrite("output.jpg", panorama);

	waitKey(0);

	return 0;
}
