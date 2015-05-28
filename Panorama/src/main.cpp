#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <list>
#include <set>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Scene.h"
#include "ImageMatching.h"
#include "Calibration.h"

using namespace std;
using namespace cv;

int composePanorama(int setId)
{
	initModule_features2d();
	initModule_nonfree();

	vector<string> sourceImagesNames;
	string setName = "";

	switch (setId) {
		case 1:
			sourceImagesNames.push_back("balcony0");
			sourceImagesNames.push_back("balcony1");
			sourceImagesNames.push_back("balcony2");
			setName = "balcony";
			break;
		case 2:
			sourceImagesNames.push_back("office1");
			sourceImagesNames.push_back("office2");
			sourceImagesNames.push_back("office4");
			sourceImagesNames.push_back("office3");
			setName = "office";
			break;
		case 3:
			sourceImagesNames.push_back("building1");
			sourceImagesNames.push_back("building2");
			sourceImagesNames.push_back("building3");
			sourceImagesNames.push_back("building4");
			sourceImagesNames.push_back("building5");
			sourceImagesNames.push_back("building6");
			setName = "building";
			break;
		case 4:
			sourceImagesNames.push_back("mountain1");
			sourceImagesNames.push_back("mountain2");
			setName = "mountains";
			break;
		case 5:
			sourceImagesNames.push_back("cliff1");
			sourceImagesNames.push_back("cliff2");
			sourceImagesNames.push_back("cliff3");
			sourceImagesNames.push_back("cliff4");
			sourceImagesNames.push_back("cliff5");
			sourceImagesNames.push_back("cliff6");
			sourceImagesNames.push_back("cliff7");
			setName = "cliff";
			break;
		default:
			return 1;
	}

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

	float a = 1;
	int projSizeX = 1024 * a;
	int projSizeY = 512 * a;
	double focalLength = getMedianFocalLength(focalLengths);
	
	cout << "Compositing final image" << endl;
	Mat finalImage = scene.composePanoramaSpherical(projSizeX, projSizeY, focalLength);

	cout << "Writing final image" << endl;
	stringstream sstr;

	sstr << "multiband-" << setName << ".jpg";
	imwrite(sstr.str(), finalImage);
	namedWindow(sstr.str(), WINDOW_AUTOSIZE);
	imshow(sstr.str(), finalImage);

	cout << "Done" << endl;

	return 0;
}

int main(int argc, char *argv[])
{
	/*for (int i = 1; i <= 6; ++i) {
		composePanorama(i);
		waitKey(1);
	}*/

	composePanorama(3);

	waitKey(0);
}
