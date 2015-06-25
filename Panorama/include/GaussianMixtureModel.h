#ifndef GAUSSIAN_MIXTURE_MODEL_H_
#define GAUSSIAN_MIXTURE_MODEL_H_

#include <vector>

#include <opencv2/core/core.hpp>

struct GaussianDistribution {
	cv::Vec3d mean;
	double deviation;
	double weight;
};

class GaussianMixture {
public:
	GaussianMixture(int nbFrames = 0) : _nbFrames(nbFrames), _nbProcessedFrames(0) {}

	void update(cv::Vec3d color);
	void normalize();
	int getNbDistributions() const;
	int getNbBackground() const;
	cv::Vec3d getBackgroundColor(int B) const;
	const GaussianDistribution &getDistribution(int i) const;

private:
	int _nbFrames;
	int _nbProcessedFrames;
	std::vector<GaussianDistribution> _mixture;
};

#endif
