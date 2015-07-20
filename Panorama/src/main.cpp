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
#include "ImageSequence.h"
#include "MatchGraph.h"
#include "Calibration.h"
#include "Scene.h"
#include "Configuration.h"

using namespace std;
using namespace cv;

int composePanorama(const string &configFilename)
{
	initModule_features2d();
	initModule_nonfree();

	Configuration::getInstance()->loadConfig(configFilename);

	vector<string> sourceImagesNames;
	string baseName = "../moving_camera_datasets/people1/input_";
	int nbImagesDataset = 30;

	for (int i = 0; i < nbImagesDataset; i += 1) {
		stringstream sstr;

		sstr << baseName << setfill('0') << setw(4) << (i + 1) << ".jpg";
		sourceImagesNames.push_back(sstr.str());
	}

	int nbImages = sourceImagesNames.size();
	ImagesRegistry images;
	clock_t start = clock();

	for (int i = 0; i < nbImages; ++i) {
		cout << "\rOpening images " << (i + 1) << "/" << nbImages << flush;
		if (!images.addImage(sourceImagesNames[i])) {
			cerr << "Error when opening image " << sourceImagesNames[i] << endl;
			return 1;
		}
	}

	cout << endl;
	images.extractFeatures();
	cout << "Feature extraction: " << (static_cast<float>(clock() - start) / CLOCKS_PER_SEC) << "s" << endl;
	start = clock();

	ImageSequence sequence;

	for (int i = 0; i < nbImages; ++i) {
		cout << "\rSequencing images " << (i + 1) << "/" << nbImages << flush;
		sequence.addImage(i, images);
	}

	cout << endl;
	cout << "Sequencing: " << (static_cast<float>(clock() - start) / CLOCKS_PER_SEC) << "s" << endl;

	ImagesRegistry images2;

	for (int i = 0; i < sequence.getNbKeyframes(); ++i) {
		images2.addImage(sourceImagesNames[sequence.getKeyFrame(i)]);
	}

	images2.extractFeatures();

	MatchGraph graph(images2);
	vector<Scene> scenes;

	cout << "Creating scenes" << endl;
	graph.createScenes(scenes, sequence);

	assert(scenes.size() == 1);

	sequence.addIntermediateFramesToScene(scenes[0]);
	scenes[0].setEstimatedFocalLength(sequence.estimateFocalLength());
	
	cout << scenes.size() << " scenes built" << endl;

	for (size_t i = 0; i < scenes.size(); ++i) {
		cout << "Compositing final image " << i << endl;
		Mat finalImage = scenes[i].composePanorama(images);

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
	if (argc < 2) {
		cerr << "Config file required" << endl;
		return 1;
	}

	composePanorama(string(argv[1]));
	waitKey(0);
	cin.get();

	return 0;
}
