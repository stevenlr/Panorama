#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <list>
#include <set>
#include <sstream>
#include <ctime>
#include <algorithm>
#include <ctime>
#include <iomanip>

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

int composePanorama(bool shuffle)
{
	initModule_features2d();
	initModule_nonfree();

	vector<string> sourceImagesNames;
	string baseName = "../moving_camera_datasets/skating/input_";
	int nbImagesDataset = 190;

	for (int i = 0; i < nbImagesDataset; i += 15) {
		stringstream sstr;

		sstr << baseName << setfill('0') << setw(4) << (i + 1) << ".jpg";
		sourceImagesNames.push_back(sstr.str());
	}

	if (shuffle) {
		random_shuffle(sourceImagesNames.begin(), sourceImagesNames.end());
	}

	int nbImages = sourceImagesNames.size();
	ImagesRegistry images;

	float featureExtractionTimeTotal = 0;

	for (int i = 0; i < nbImages; ++i) {
		cout << "\rReading images and extracting features " << (i + 1) << "/" << nbImages << flush;

		Mat img = imread(sourceImagesNames[i]);

		if (!img.data) {
			cerr << "Error when opening image " << sourceImagesNames[i] << endl;
			return 1;
		}

		clock_t start = clock();
		images.addImage(img);
		featureExtractionTimeTotal += static_cast<float>(clock() - start) / CLOCKS_PER_SEC;
	}

	cout << endl;

	featureExtractionTimeTotal /= nbImages;
	cout << "Feature extraction average: " << featureExtractionTimeTotal << "s" << endl;

	MatchGraph graph(images);
	vector<Scene> scenes;

	cout << "Creating scenes" << endl;
	graph.createScenes(scenes);

	float width = 2048;
	int projSizeX = static_cast<int>(width);
	int projSizeY = static_cast<int>(width / 2);
	
	cout << scenes.size() << " scenes built" << endl;

	for (size_t i = 0; i < scenes.size(); ++i) {
		cout << "Compositing final image " << i << endl;
		Mat finalImage = scenes[i].composePanoramaSpherical(images, projSizeX, projSizeY);

		if (finalImage.size() == Size(0, 0)) {
			continue;
		}

		stringstream sstr;

		sstr << "output-" << i << ".jpg";
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
	composePanorama(true);
	waitKey(0);

	return 0;
}
