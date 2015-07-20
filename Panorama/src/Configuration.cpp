#include "Configuration.h"

#include <iostream>
#include <stdexcept>

#include "tinyxml2.h"

using namespace std;
using namespace tinyxml2;

Configuration *Configuration::instance = nullptr;

Configuration *Configuration::getInstance()
{
	if (!instance) {
		instance = new Configuration();
	}

	return instance;
}

Configuration::Configuration()
{
	keyframeOverlapThreshold = 0.75;
	featureMatchConfidence = 0.5;
	registrationOptimizationIterations = 50;
	imageMatchConfidenceThreshold = 1.0;
	nbClusters = 4;
	clusteringStdDevThreshold = 21;
	madForegroundThreshold = 3;
}

void Configuration::loadConfig(const std::string &filename)
{
	XMLDocument doc;

	if (doc.LoadFile(filename.c_str())) {
		throw exception("Error when opening config file");
	}

	XMLElement *elmt = doc.FirstChildElement("configuration");

	elmt->FirstChildElement("keyframeOverlapThreshold")->QueryFloatText(&keyframeOverlapThreshold);
	elmt->FirstChildElement("featureMatchConfidence")->QueryFloatText(&featureMatchConfidence);
	elmt->FirstChildElement("registrationOptimizationIterations")->QueryIntText(&registrationOptimizationIterations);
	elmt->FirstChildElement("imageMatchConfidenceThreshold")->QueryFloatText(&imageMatchConfidenceThreshold);
	elmt->FirstChildElement("nbClusters")->QueryIntText(&nbClusters);
	elmt->FirstChildElement("clusteringStdDevThreshold")->QueryFloatText(&clusteringStdDevThreshold);
	elmt->FirstChildElement("madForegroundThreshold")->QueryFloatText(&madForegroundThreshold);
}

float Configuration::getKeyframeOverlapThreshold() const
{
	return keyframeOverlapThreshold;
}

float Configuration::getFeatureMatchConfidence() const
{
	return featureMatchConfidence;
}

int Configuration::getRegistrationOptimizationIterations() const
{
	return registrationOptimizationIterations;
}

float Configuration::getImageMatchConfidenceThreshold() const
{
	return imageMatchConfidenceThreshold;
}

int Configuration::getNbClusters() const
{
	return nbClusters;
}

float Configuration::getClusteringStdDevThreshold() const
{
	return clusteringStdDevThreshold;
}

float Configuration::getMadForegroundThreshold() const
{
	return madForegroundThreshold;
}
