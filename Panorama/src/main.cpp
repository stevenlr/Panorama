#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define NB_IMAGES 2

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
	initModule_features2d();
	initModule_nonfree();

	string sourceImagesNames[NB_IMAGES];
	Mat sourceImages[NB_IMAGES];

	sourceImagesNames[0] = "bus1";
	sourceImagesNames[1] = "bus2";

	for (int i = 0; i < NB_IMAGES; ++i) {
		sourceImages[i] = imread("../source_images/" + sourceImagesNames[i] + ".jpg");

		if (!sourceImages[i].data) {
			cout << "Error when opening image " << sourceImagesNames[i] << endl;
		}

		stringstream str;
		str << "source " << sourceImagesNames[i];

		namedWindow(str.str());
		imshow(str.str(), sourceImages[i]);
	}

	Ptr<FeatureDetector> featureDetector = FeatureDetector::create("SIFT");
	Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create("SIFT");

	vector<KeyPoint> keypoints[NB_IMAGES];
	Mat featureDescriptors[NB_IMAGES];

	for (int i = 0; i < NB_IMAGES; ++i) {
		featureDetector->detect(sourceImages[i], keypoints[i]);
		descriptorExtractor->compute(sourceImages[i], keypoints[i], featureDescriptors[i]);
	}

	const int image1 = 0;
	const int image2 = 1;

	Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create("FlannBased");
	vector<DMatch> matches;

	descriptorMatcher->match(featureDescriptors[image2], featureDescriptors[image1], matches);

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

	Mat matchImage;

	drawMatches(sourceImages[image2], keypoints[image2],
				sourceImages[image1], keypoints[image1],
				goodMatches, matchImage,
				Scalar::all(-1), Scalar::all(-1), vector<char>(),
				DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	namedWindow("matches");
	imshow("matches", matchImage);

	vector<Point2f> points[2];

	for (int i = 0; i < goodMatches.size(); ++i) {
		points[0].push_back(keypoints[image2][goodMatches[i].queryIdx].pt);
		points[1].push_back(keypoints[image1][goodMatches[i].trainIdx].pt);
	}

	Mat homography = findHomography(points[0], points[1], CV_RANSAC);
	Mat transformedImage;

	warpPerspective(sourceImages[image2], transformedImage, homography, sourceImages[image2].size());

	namedWindow("transformed");
	imshow("transformed", transformedImage);

	waitKey(0);

	return 0;
}