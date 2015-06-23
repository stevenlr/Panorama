#include "GaussianMixtureModel.h"

#include <iostream>

using namespace std;
using namespace cv;

namespace {
	bool compareGaussianDistribution(const GaussianDistribution &first, const GaussianDistribution &second)
	{
		return first.weight < second.weight;
	}
}

void GaussianMixture::update(Vec3d color)
{
	const double deviation0 = 4;
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

		if (malDist < 2 * distrib.deviation) {
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
}

void GaussianMixture::normalize()
{
	list<GaussianDistribution>::iterator it = _mixture.begin();
	double sum = 0;

	while (it != _mixture.end()) {
		sum += (*it++).weight;
	}

	it = _mixture.begin();

	while (it != _mixture.end()) {
		(*it++).weight /= sum;
	}
}

int GaussianMixture::getNbDistributions() const
{
	return _mixture.size();
}

int GaussianMixture::getNbBackground() const
{
	const double cf = 0.1;
	int bMax = -1;
	double sumMax = 0;
	int nbDistrib = getNbDistributions();

	for (int i = 0; i < nbDistrib; ++i) {
		list<GaussianDistribution>::const_iterator it = _mixture.cbegin();
		int j = 0;
		double sum = 0;

		while (it != _mixture.cend() && j <= i) {
			sum += (*it++).weight;
			j++;
		}

		if (sum > (1 - cf)) {
			if (sum > sumMax) {
				sumMax = sum;
				bMax = i;
			}
		}
	}

	return bMax;
}

Vec3d GaussianMixture::getBackgroundColor(int B) const
{
	Vec3d color(0, 0, 0);
	int nbDistrib = getNbDistributions();
	list<GaussianDistribution>::const_iterator it = _mixture.cbegin();

	for (int i = 0; i <= B; ++i) {
		const GaussianDistribution &distrib = *it++;

		color += distrib.weight * distrib.mean;
	}

	return color;
}
