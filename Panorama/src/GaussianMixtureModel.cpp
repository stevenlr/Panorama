#include "GaussianMixtureModel.h"

using namespace std;
using namespace cv;

namespace {
	bool compareGaussianDistribution(const GaussianDistribution &first, const GaussianDistribution &second)
	{
		return first.weight > second.weight;
	}
}

void GaussianMixture::update(Vec3d color)
{
	const double deviation0 = 6;
	double alpha = 1.0 / _nbFrames;

	if (_nbFrames == 0) {
		return;
	}

	if (_mixture.size() == 0) {
		GaussianDistribution distrib;

		distrib.mean = color;
		distrib.deviation = deviation0;
		distrib.weight = alpha;
		_mixture.push_back(distrib);

		return;
	}

	list<GaussianDistribution>::iterator it = _mixture.begin();
	bool isCloseToADistrib = false;

	while (it != _mixture.end() && !isCloseToADistrib) {
		GaussianDistribution &distrib = *it++;
		Vec3d diffColor = color - distrib.mean;
		double colorDistSquared = diffColor.dot(diffColor);
		double malDist = std::sqrt(colorDistSquared) / distrib.deviation;

		if (malDist > 3 * distrib.deviation) {
			isCloseToADistrib = true;
			distrib.weight = distrib.weight + alpha * (1 - distrib.weight);
			distrib.mean = distrib.mean + diffColor * alpha / distrib.weight;
			distrib.deviation = std::sqrt(distrib.deviation * distrib.deviation + alpha / distrib.weight * (colorDistSquared - distrib.deviation * distrib.deviation));
		} else {
			distrib.weight = distrib.weight - alpha * distrib.weight;
		}
	}

	if (!isCloseToADistrib) {
		GaussianDistribution distrib;

		distrib.mean = color;
		distrib.deviation = deviation0;
		distrib.weight = alpha;
		_mixture.push_back(distrib);
	}

	_mixture.sort(compareGaussianDistribution);
	// prune ?
}
