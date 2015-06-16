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

using namespace std;
using namespace cv;

/*int composePanorama(bool shuffle)
{
	initModule_features2d();
	initModule_nonfree();

	vector<string> sourceImagesNames;
	string baseName = "../moving_camera_datasets/skating/input_";
	int nbImagesDataset = 190;

	for (int i = 0; i < nbImagesDataset; i += 1) {
		stringstream sstr;

		sstr << baseName << setfill('0') << setw(4) << (i + 1) << ".jpg";
		sourceImagesNames.push_back(sstr.str());
	}

	/*sourceImagesNames.push_back("../source_images/building1.jpg");
	sourceImagesNames.push_back("../source_images/building2.jpg");
	sourceImagesNames.push_back("../source_images/building3.jpg");
	sourceImagesNames.push_back("../source_images/building4.jpg");
	sourceImagesNames.push_back("../source_images/building5.jpg");
	sourceImagesNames.push_back("../source_images/building6.jpg");
	sourceImagesNames.push_back("../source_images/office1.jpg");
	sourceImagesNames.push_back("../source_images/office2.jpg");
	sourceImagesNames.push_back("../source_images/office3.jpg");
	sourceImagesNames.push_back("../source_images/office4.jpg");
	sourceImagesNames.push_back("../source_images/balcony0.jpg");
	sourceImagesNames.push_back("../source_images/balcony1.jpg");
	sourceImagesNames.push_back("../source_images/balcony2.jpg");
	sourceImagesNames.push_back("../source_images/cliff1.jpg");
	sourceImagesNames.push_back("../source_images/cliff2.jpg");
	sourceImagesNames.push_back("../source_images/cliff3.jpg");
	sourceImagesNames.push_back("../source_images/cliff4.jpg");
	sourceImagesNames.push_back("../source_images/cliff5.jpg");
	sourceImagesNames.push_back("../source_images/cliff6.jpg");
	sourceImagesNames.push_back("../source_images/cliff7.jpg");
	sourceImagesNames.push_back("../source_images/mountain1.jpg");
	sourceImagesNames.push_back("../source_images/mountain2.jpg");*//*

	if (shuffle) {
		random_shuffle(sourceImagesNames.begin(), sourceImagesNames.end());
	}

	int nbImages = sourceImagesNames.size();
	ImagesRegistry images;

	float featureExtractionTimeTotal = 0;

	for (int i = 0; i < nbImages; ++i) {
		cout << "\rReading images and extracting features " << (i + 1) << "/" << nbImages << flush;
		clock_t start = clock();

		if (!images.addImage(sourceImagesNames[i])) {
			cerr << "Error when opening image " << sourceImagesNames[i] << endl;
			return 1;
		}

		featureExtractionTimeTotal += static_cast<float>(clock() - start) / CLOCKS_PER_SEC;
	}

	cout << endl;

	featureExtractionTimeTotal /= nbImages;
	cout << "Feature extraction average: " << featureExtractionTimeTotal << "s" << endl;

	ImageSequence sequence;

	for (int i = 0; i < nbImages; ++i) {
		sequence.addImage(i, images);
	}

	ImagesRegistry images2;

	for (int i = 0; i < sequence.getNbKeyframes(); ++i) {
		images2.addImage(sourceImagesNames[sequence.getKeyFrame(i)]);
	}

	MatchGraph graph(images2);
	vector<Scene> scenes;

	cout << "Creating scenes" << endl;
	graph.createScenes(scenes);

	float a = 1;
	int projSizeX = static_cast<int>(1024 * a);
	int projSizeY = static_cast<int>(512 * a);

	
	cout << scenes.size() << " scenes built" << endl;

	for (size_t i = 0; i < scenes.size(); ++i) {
		cout << "Compositing final image " << i << endl;
		Mat finalImage = scenes[i].composePanoramaSpherical(images2, projSizeX, projSizeY);

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
}*/

int composePanorama2()
{
	initModule_features2d();
	initModule_nonfree();

	vector<string> sourceImagesNames;
	string baseName = "../moving_camera_datasets/people1/input_";
	int nbImagesDataset = 41;

	for (int i = 0; i < nbImagesDataset; i += 1) {
		stringstream sstr;

		sstr << baseName << setfill('0') << setw(4) << (i + 1) << ".jpg";
		sourceImagesNames.push_back(sstr.str());
	}

	int nbImages = sourceImagesNames.size();
	ImagesRegistry images;

	float featureExtractionTimeTotal = 0;

	for (int i = 0; i < nbImages; ++i) {
		cout << "\rReading images and extracting features " << (i + 1) << "/" << nbImages << flush;
		clock_t start = clock();

		if (!images.addImage(sourceImagesNames[i])) {
			cerr << "Error when opening image " << sourceImagesNames[i] << endl;
			return 1;
		}

		featureExtractionTimeTotal += static_cast<float>(clock() - start) / CLOCKS_PER_SEC;
	}

	cout << endl;

	featureExtractionTimeTotal /= nbImages;
	cout << "Feature extraction average: " << featureExtractionTimeTotal << "s" << endl;

	ImageSequence sequence;

	for (int i = 0; i < nbImages; ++i) {
		cout << "\rSequencing images " << (i + 1) << "/" << nbImages << flush;
		sequence.addImage(i, images);
	}

	cout << endl;

	ImagesRegistry images2;

	for (int i = 0; i < sequence.getNbKeyframes(); ++i) {
		images2.addImage(sourceImagesNames[sequence.getKeyFrame(i)]);
	}

	MatchGraph graph(images2);
	vector<Scene> scenes;

	cout << "Creating scenes" << endl;
	graph.createScenes(scenes, sequence);

	assert(scenes.size() == 1);

	sequence.addIntermediateFramesToScene(scenes[0]);
	scenes[0].setEstimatedFocalLength(sequence.estimateFocalLength());

	float width = 2048;
	int projSizeX = static_cast<int>(width);
	int projSizeY = static_cast<int>(width / 2);

	/**
	 * Bundle adjustment :
	 *  - between every keyframe (stored in matchgraph)
	 *  - from each intermediate frame to the last keyframe (stored in imagesequence)
	 */
	
	cout << scenes.size() << " scenes built" << endl;

	for (size_t i = 0; i < scenes.size(); ++i) {
		cout << "Bundle adjustment " << i << endl;
		//scenes[i].bundleAdjustment(images, graph);

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
	composePanorama2();
	waitKey(0);
	cin.get();

	return 0;
}
