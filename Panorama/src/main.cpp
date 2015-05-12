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
#include <opencv2/stitching/detail/autocalib.hpp>

#define PI 3.14159265358979323846

#include "Scene.h"
#include "ImageMatching.h"

#define NB_IMAGES 3

using namespace std;
using namespace cv;

typedef pair<pair<int, int>, ImageMatchInfos> MatchMatrixElement;

bool compareMatchMatrixElements(const MatchMatrixElement &first, const MatchMatrixElement &second)
{
	return first.second.avgDistance < second.second.avgDistance;
}

bool checkCycle(const Scene &scene, int image)
{
	set<int> stack;
	int current = image;

	stack.insert(image);

	while (current != -1) {
		current = scene.getParent(current);

		if (stack.find(current) != stack.end()) {
			return true;
		}

		stack.insert(current);
	}

	return false;
}

void cameraPoseFromHomography(const Mat &H, Mat &pose)
{
	pose = Mat::eye(3, 4, CV_64F);

	float norm1 = (float) norm(H.col(0));  
	float norm2 = (float) norm(H.col(1));  
	float tnorm = (norm1 + norm2) / 2;

	Mat p1 = H.col(0);
	Mat p2 = pose.col(0);

	normalize(p1, p2);

	p1 = H.col(1);
	p2 = pose.col(1);

	normalize(p1, p2);

	p1 = pose.col(0);
	p2 = pose.col(1);

	Mat p3 = p1.cross(p2);
	Mat c2 = pose.col(2);

	p3.copyTo(c2);
	pose.col(3) = H.col(2) / tnorm;
}

void findFocalLength(const Mat &homography, vector<double> &focalLengths)
{
	double f0, f1;
	bool ok0, ok1;

	detail::focalsFromHomography(homography, f0, f1, ok0, ok1);

	if (ok0 && ok1) {
		focalLengths.push_back(sqrt(f0 * f1));
	}
}

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

			matchMatrix.push_back(make_pair(make_pair(i, j), matchInfos));
			matchMatrix.push_back(make_pair(make_pair(j, i), matchInfos));
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

		if (checkCycle(scene, objectImage)) {
			scene.setParent(objectImage, -1);
			matchMatrix.erase(matchMatrix.begin());
			continue;
		}

		Mat homography = computeHomography(descriptors[sceneImage], descriptors[objectImage], elt.second);
		Mat pose;

		cameraPoseFromHomography(homography, pose);

		double rx, ry, rz;

		if (abs(pose.at<double>(2, 0)) != 1) {
			double y1 = -asin(pose.at<double>(2, 0));
			double y2 = PI - y1;

			double x1 = atan2(pose.at<double>(2, 1) / cos(y1), pose.at<double>(2, 2) / cos(y1));
			double x2 = atan2(pose.at<double>(2, 1) / cos(y2), pose.at<double>(2, 2) / cos(y2));

			double z1 = atan2(pose.at<double>(1, 0) / cos(y1), pose.at<double>(1, 1) / cos(y1));
			double z2 = atan2(pose.at<double>(1, 0) / cos(y2), pose.at<double>(1, 1) / cos(y2));

			if (abs(x1) < abs(x2)) {
				rx = x1;
				ry = y1;
				rz = z1;
			} else {
				rx = x2;
				ry = y2;
				rz = z2;
			}
		} else {
			rz = 0;

			if (pose.at<double>(2, 0) == -1) {
				ry = PI / 2;
				rz = atan2(pose.at<double>(0, 1), pose.at<double>(0, 2));
			} else {
				ry = -PI / 2;
				rz = atan2(-pose.at<double>(0, 1), -pose.at<double>(0, 2));
			}
		}

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

	double focalLength;
	int numFocalLengths = focalLengths.size();

	std::sort(focalLengths.begin(), focalLengths.end());

	if (focalLengths.size() % 2 == 0) {
		focalLength = (focalLengths[numFocalLengths / 2 - 1] + focalLengths[numFocalLengths / 2]) / 2;
	} else {
		focalLength = focalLengths[numFocalLengths / 2];
	}

	cout << focalLength << endl;

	Mat panorama = scene.composePanorama();

	namedWindow("panorama", WINDOW_AUTOSIZE);
	imshow("panorama", panorama);

	imwrite("output.jpg", panorama);

	waitKey(0);

	return 0;
}
