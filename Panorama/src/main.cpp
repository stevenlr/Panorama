#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <list>
#include <set>
#include <sstream>
#include <ctime>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "ImagesRegistry.h"
#include "MatchGraph.h"
#include "Calibration.h"
#include "Scene.h"

using namespace std;
using namespace cv;

int composePanorama(int setId)
{
	initModule_features2d();
	initModule_nonfree();

	vector<string> sourceImagesNames;

	switch (setId) {
		case 1:
			sourceImagesNames.push_back("balcony0");
			sourceImagesNames.push_back("balcony1");
			sourceImagesNames.push_back("balcony2");
			sourceImagesNames.push_back("office1");
			sourceImagesNames.push_back("office2");
			sourceImagesNames.push_back("office4");
			sourceImagesNames.push_back("office3");
			sourceImagesNames.push_back("mountain1");
			sourceImagesNames.push_back("mountain2");
			sourceImagesNames.push_back("building1");
			sourceImagesNames.push_back("building2");
			sourceImagesNames.push_back("building3");
			sourceImagesNames.push_back("building4");
			sourceImagesNames.push_back("building5");
			sourceImagesNames.push_back("building6");
			sourceImagesNames.push_back("cliff1");
			sourceImagesNames.push_back("cliff2");
			sourceImagesNames.push_back("cliff3");
			sourceImagesNames.push_back("cliff4");
			sourceImagesNames.push_back("cliff5");
			sourceImagesNames.push_back("cliff6");
			sourceImagesNames.push_back("cliff7");
			random_shuffle(sourceImagesNames.begin(), sourceImagesNames.end());
			//random_shuffle(sourceImagesNames.begin(), sourceImagesNames.end());
			break;
		case 2:
			sourceImagesNames.push_back("office1");
			sourceImagesNames.push_back("office2");
			sourceImagesNames.push_back("office4");
			sourceImagesNames.push_back("office3");
			break;
		case 3:
			sourceImagesNames.push_back("building1");
			sourceImagesNames.push_back("building2");
			sourceImagesNames.push_back("building3");
			sourceImagesNames.push_back("building4");
			sourceImagesNames.push_back("building5");
			sourceImagesNames.push_back("building6");
			break;
		case 4:
			sourceImagesNames.push_back("mountain1");
			sourceImagesNames.push_back("mountain2");
			break;
		case 5:
			sourceImagesNames.push_back("cliff1");
			sourceImagesNames.push_back("cliff2");
			sourceImagesNames.push_back("cliff3");
			sourceImagesNames.push_back("cliff4");
			sourceImagesNames.push_back("cliff5");
			sourceImagesNames.push_back("cliff6");
			sourceImagesNames.push_back("cliff7");
			break;
		default:
			return 1;
	}

	int nbImages = sourceImagesNames.size();
	ImagesRegistry images;

	for (int i = 0; i < nbImages; ++i) {
		cout << "\rReading images and extracting features " << (i + 1) << "/" << nbImages << flush;

		Mat img = imread("../source_images/" + sourceImagesNames[i] + ".jpg");

		if (!img.data) {
			cerr << "Error when opening image " << sourceImagesNames[i] << endl;
			return 1;
		}

		images.addImage(img);
	}

	cout << endl;

	MatchGraph graph(images);
	vector<Scene> scenes;

	cout << "Creating scenes" << endl;
	graph.createScenes(scenes);

	float a = 1;
	int projSizeX = 1024 * a;
	int projSizeY = 512 * a;
	
	cout << scenes.size() << " scenes built" << endl;

	for (int i = 0; i < scenes.size(); ++i) {
		cout << "Compositing final image " << i << endl;
		Mat finalImage = scenes[i].composePanoramaSpherical(images, projSizeX, projSizeY);

		if (finalImage.size() == Size(0, 0)) {
			continue;
		}

		stringstream sstr;

		sstr << "output-" << sourceImagesNames[0] << "-" << i << ".jpg";
		imwrite(sstr.str(), finalImage);
		namedWindow(sstr.str(), WINDOW_AUTOSIZE);
		imshow(sstr.str(), finalImage);
		waitKey(1);
	}

	cout << "Done" << endl;

	return 0;
}

int main(int argc, char *argv[])
{
	/*for (int i = 2; i <= 6; ++i) {
		composePanorama(i);
		waitKey(1);
	}*/

	composePanorama(1);

	waitKey(0);
}
