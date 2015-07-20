#ifndef CONFIGURATION_H_
#define CONFIGURATION_H_

#include <string>

class Configuration {
public:
	static Configuration *getInstance();
	void loadConfig(const std::string &filename);

	float getKeyframeOverlapThreshold() const;
	float getFeatureMatchConfidence() const;
	int getRegistrationOptimizationIterations() const;
	float getImageMatchConfidenceThreshold() const;
	int getNbClusters() const;
	float getClusteringStdDevThreshold() const;
	float getMadForegroundThreshold() const;

private:
	Configuration();

	static Configuration *instance;

	float keyframeOverlapThreshold;
	float featureMatchConfidence;
	int registrationOptimizationIterations;
	float imageMatchConfidenceThreshold;
	int nbClusters;
	float clusteringStdDevThreshold;
	float madForegroundThreshold;
};

#endif
