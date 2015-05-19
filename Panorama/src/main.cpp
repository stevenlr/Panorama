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

	cout << "Reading images" << endl;

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

	cout << "Computing scene graph" << endl;
	scene.makeSceneGraph(matchGraphEdges, matchInfosMap);

	int projSizeX = 1024;
	int projSizeY = 512;
	double focalLength = getMedianFocalLength(focalLengths);
	
	cout << "Compositing final image" << endl;
	Mat finalImage = scene.composePanoramaSpherical(projSizeX, projSizeY, focalLength);

	cout << "Writing final image" << endl;
	namedWindow("output spherical", WINDOW_AUTOSIZE);
	imshow("output spherical", finalImage);
	imwrite("output.jpg", finalImage);

	//Mat panorama = scene.composePanoramaPlanar();

	/*namedWindow("output", WINDOW_AUTOSIZE);
	imshow("output", panorama);
	imwrite("output.jpg", panorama);*/

	cout << "Done" << endl;
	waitKey(0);

	return 0;
}
