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
#include <stdexcept>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <tinyxml2.h>

#include "ImagesRegistry.h"
#include "ImageSequence.h"
#include "MatchGraph.h"
#include "Calibration.h"
#include "Scene.h"
#include "Configuration.h"

using namespace std;
using namespace cv;
using namespace tinyxml2;

void loadFileList(const string &filelistFilename, vector<string> &filelist)
{
	XMLDocument doc;

	if (doc.LoadFile(filelistFilename.c_str())) {
		throw exception("Failed to open file list");
	}

	XMLElement *root = doc.FirstChildElement("filelist");

	string folder = root->FirstChildElement("folder")->GetText();
	string prefix = root->FirstChildElement("prefix")->GetText();
	string extension = root->FirstChildElement("extension")->GetText();
	int startFrame, endFrame, frameStep, band;
	stringstream sstr;

	root->FirstChildElement("startFrame")->QueryIntText(&startFrame);
	root->FirstChildElement("endFrame")->QueryIntText(&endFrame);
	root->FirstChildElement("frameStep")->QueryIntText(&frameStep);
	root->FirstChildElement("band")->QueryIntText(&band);

	for (int frame = startFrame; frame <= endFrame; frame += frameStep) {
		sstr.str("");
		sstr << folder << "/" << prefix << setw(band) << setfill('0') << frame << "." << extension;
		filelist.push_back(sstr.str());
	}
}

int composePanorama(const string &configFilename, const string &filelistFilename)
{
	vector<string> sourceImagesNames;

	initModule_features2d();
	initModule_nonfree();
	Configuration::getInstance()->loadConfig(configFilename);
	loadFileList(filelistFilename, sourceImagesNames);

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
	if (argc < 3) {
		cerr << "Usage: " << argv[0] << " {config}.xml {filelist}.xml" << endl;
		return 1;
	}

	composePanorama(string(argv[1]), string(argv[2]));
	waitKey(0);
	cin.get();

	return 0;
}
