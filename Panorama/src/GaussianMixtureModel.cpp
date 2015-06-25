#include "GaussianMixtureModel.h"

#include <iostream>
#include <algorithm>

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
	const double deviation0 = 10;
	double alpha = 1.0 / (_nbProcessedFrames + 1);

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

	vector<GaussianDistribution>::iterator it = _mixture.begin();
	bool isCloseToADistrib = false;

	while (it != _mixture.end() && !isCloseToADistrib) {
		GaussianDistribution &distrib = *it++;
		Vec3d diffColor = color - distrib.mean;
		double colorDist = max(max(abs(diffColor[0]), abs(diffColor[1])), abs(diffColor[2]));
		double colorDistSquared = colorDist * colorDist;
		double malDist = std::sqrt(colorDistSquared) / distrib.deviation;

		if (malDist < 3 * distrib.deviation) {
			isCloseToADistrib = true;
			distrib.mean = distrib.mean + diffColor * alpha / distrib.weight;
			distrib.deviation = std::sqrt(distrib.deviation * distrib.deviation + alpha / distrib.weight * (colorDistSquared - distrib.deviation * distrib.deviation));
			distrib.weight = distrib.weight + alpha * (1 - distrib.weight);
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

	std::sort(_mixture.begin(), _mixture.end(), compareGaussianDistribution);
	normalize();
	_nbProcessedFrames++;
}

void GaussianMixture::normalize()
{
	vector<GaussianDistribution>::iterator it = _mixture.begin();
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
	const double cf = 0.01;
	int b = -1;
	double sumMin = numeric_limits<double>::max();
	int nbDistrib = getNbDistributions();

	for (int i = 0; i < nbDistrib; ++i) {
		vector<GaussianDistribution>::const_iterator it = _mixture.cbegin();
		int j = 0;
		double sum = 0;

		while (it != _mixture.cend() && j <= i) {
			sum += (*it++).weight;
			j++;
		}

		if (sum > (1 - cf)) {
			if (sum < sumMin) {
				sumMin = sum;
				b = i;
			}
		}
	}

	return b;
}

Vec3d GaussianMixture::getBackgroundColor(int B) const
{
	Vec3d color(0, 0, 0);
	int nbDistrib = getNbDistributions();
	vector<GaussianDistribution>::const_iterator it = _mixture.cbegin();
	double sum = 0;

	//int num = 2;

	for (int i = 0; i <= B; ++i) {
		const GaussianDistribution &distrib = *it++;

		//if (num == i) {
			color += distrib.weight * distrib.mean;
			sum += distrib.weight;
		//}
	}

	return color / sum;
}

const GaussianDistribution &GaussianMixture::getDistribution(int i) const
{
	return _mixture[i];
}
