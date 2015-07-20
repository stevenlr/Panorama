#include "Scene.h"

#include <set>
#include <queue>
#include <algorithm>
#include <iostream>
#include <ctime>
#include <iomanip>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Constants.h"
#include "Calibration.h"
#include "Configuration.h"

using namespace std;
using namespace cv;

Scene::Scene()
{
	_estimatedFocalLength = -1;
	_nbImages = 0;
}

void Scene::setEstimatedFocalLength(double f)
{
	_estimatedFocalLength = f;
}

void Scene::addImage(int id)
{
	_reverseIds.insert(make_pair(id, _nbImages++));
	_images.push_back(id);
	_transform.push_back(Mat());
	_parent.push_back(-1);
	_cameras.push_back(Camera());
}

int Scene::getNbImages() const
{
	return _nbImages;
}

int Scene::getImage(int id) const
{
	assert(id >= 0 && id < _nbImages);
	return _images[id];
}

int Scene::getIdInScene(int image) const
{
	map<int, int>::const_iterator it = _reverseIds.find(image);

	if (it == _reverseIds.cend()) {
		return -1;
	}

	return it->second;
}

int Scene::getParent(int image) const
{
	return _parent[image];
}

void Scene::setParent(int image, int parent)
{
	assert(image >= 0 && image < _nbImages);
	assert(parent >= -1 && parent < _nbImages);
	assert(image != parent);

	_parent[image] = parent;
}

void Scene::setTransform(int image, const Mat &transform)
{
	assert(image >= 0 && image < _nbImages);
	assert(transform.size() == Size(3, 3));

	_transform[image] = transform;
}

const cv::Mat &Scene::getTransform(int image) const
{
	assert(image >= 0 && image < _nbImages);

	return _transform[image];
}

Mat Scene::getFullTransform(int image) const
{
	assert(image >= 0 && image < _nbImages);

	if (_parent[image] != -1) {
		return getFullTransform(_parent[image]) * _transform[image];
	} else {
		return _transform[image].clone();
	}
}

namespace {
	pair<Point2i, Point2i> getOverlappingRegion(const pair<Point2i, Point2i> &a, const pair<Point2i, Point2i> &b)
	{
		Point2i maxCorner, minCorner;

		minCorner.x = std::max(a.first.x, b.first.x);
		minCorner.y = std::max(a.first.y, b.first.y);

		maxCorner.x = std::min(a.second.x, b.second.x);
		maxCorner.y = std::min(a.second.y, b.second.y);

		return make_pair(minCorner, maxCorner);
	}

	float getWeight(int x, int size) {
		return 1 - std::abs((static_cast<float>(x) / size) * 2 - 1);
	}
}

int Scene::getRootNode() const
{
	int parent, node = 0;

	while ((parent = getParent(node)) != -1) {
		node = parent;
	}

	return node;
}

bool compareVec3f(Vec3f first, Vec3f second)
{
	float d1 = first.dot(first);
	float d2 = second.dot(second);

	return d1 < d2;
}

Mat Scene::composePanorama(const ImagesRegistry &images)
{
	const int nbClusters = Configuration::getInstance()->getNbClusters();
	const float minStdDev = Configuration::getInstance()->getClusteringStdDevThreshold();
	const float madForegroundThreshold = Configuration::getInstance()->getMadForegroundThreshold();
	vector<Mat> warpedImages(_nbImages);
	vector<Mat> warpedMasks(_nbImages);
	vector<pair<Point2d, Point2d>> corners(_nbImages);
	Point finalMinCorner(numeric_limits<int>::max(), numeric_limits<int>::max());
	Point finalMaxCorner(numeric_limits<int>::min(), numeric_limits<int>::min());
	clock_t start;
	float elapsedTime;

	if (_nbImages < 2) {
		return Mat();
	}

	start = clock();
	for (int i = 0; i < _nbImages; ++i) {
		Mat img = images.getImage(getImage(i));
		Size size = img.size();
		Mat homography = getFullTransform(i).clone();
		Point2i minCorner(numeric_limits<int>::max(), numeric_limits<int>::max());
		Point2i maxCorner(numeric_limits<int>::min(), numeric_limits<int>::min());
		Mat translation = Mat::eye(Size(3, 3), CV_64F);

		if (homography.size() != Size(3, 3)) {
			continue;
		}

		cout << "\r  Warping images " << (i + 1) << "/" << _nbImages << flush;

		translation.at<double>(0, 2) = -size.width / 2;
		translation.at<double>(1, 2) = -size.height / 2;
		homography = translation.inv() * homography * translation;

		for (int y = 0; y < size.height; y += size.height - 1) {
			for (int x = 0; x < size.width; x += size.width - 1) {
				Mat_<double> point = Mat_<double>::ones(Size(1, 3));

				point(0, 0) = x;
				point(1, 0) = y;

				point = homography * point;

				int px = static_cast<int>(point(0, 0) / point(2, 0));
				int py = static_cast<int>(point(1, 0) / point(2, 0));

				minCorner.x = std::min(minCorner.x, px);
				minCorner.y = std::min(minCorner.y, py);
				maxCorner.x = std::max(maxCorner.x, px);
				maxCorner.y = std::max(maxCorner.y, py);
			}
		}

		translation.at<double>(0, 2) = -minCorner.x;
		translation.at<double>(1, 2) = -minCorner.y;
		homography = translation * homography;

		Mat maskNormal(size, CV_8U, Scalar(255));
		Size warpedSize(maxCorner.x - minCorner.x, maxCorner.y - minCorner.y);

		warpPerspective(img, warpedImages[i], homography, warpedSize);
		warpPerspective(maskNormal, warpedMasks[i], homography, warpedSize);
		corners[i] = make_pair(minCorner, maxCorner);

		finalMinCorner.x = std::min(finalMinCorner.x, minCorner.x);
		finalMinCorner.y = std::min(finalMinCorner.y, minCorner.y);
		finalMaxCorner.x = std::max(finalMaxCorner.x, maxCorner.x);
		finalMaxCorner.y = std::max(finalMaxCorner.y, maxCorner.y);
	}

	cout << endl;

	elapsedTime = static_cast<float>(clock() - start) / CLOCKS_PER_SEC;
	cout << "  warping total: " << elapsedTime << "s" << endl;
	cout << "  writing video file" << endl;

	Size finalImageSize(finalMaxCorner.x - finalMinCorner.x + 1, finalMaxCorner.y - finalMinCorner.y + 1);
	Mat finalImage(finalImageSize, images.getImage(getImage(0)).type());
	Mat_<float> madImage(finalImageSize);
	Mat compositeImage(finalImage.size(), CV_32FC3, Scalar(0, 0, 0));

	VideoWriter videoWriter("output.avi", CV_FOURCC('M','J','P','G'), 4, finalImageSize);

	for (int i = 0; i < _nbImages; ++i) {
		finalImage.setTo(0);
		warpedImages[i].copyTo(finalImage.colRange(static_cast<int>(corners[i].first.x - finalMinCorner.x),
													static_cast<int>(corners[i].second.x - finalMinCorner.x))
										 .rowRange(static_cast<int>(corners[i].first.y - finalMinCorner.y),
													static_cast<int>(corners[i].second.y - finalMinCorner.y)));
		videoWriter.write(finalImage);

		stringstream sstr;

		sstr << "cube_" << setfill('0') << setw(4) << (i + 1) << ".png";
		imwrite(sstr.str(), finalImage);
	}

	videoWriter.release();
	finalImage.setTo(0);

	int nbPixel = finalImageSize.width * finalImageSize.height;
	TermCriteria criteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 10, 1.0);
	int *samplesCount = new int[nbClusters];

	start = clock();

	for (int y = 0; y < finalImageSize.height; ++y) {
		cout << "\r  Modeling background " << static_cast<int>(static_cast<float>(y + 1) / finalImageSize.height * 100) << "%" << flush;

		for (int x = 0; x < finalImageSize.width; ++x) {
			Mat labels, centers;
			int nbValues = 0;
			int px = x + finalMinCorner.x;
			int py = y + finalMinCorner.y;

			for (int i = 0; i < _nbImages; ++i) {
				int ix = static_cast<int>(px - corners[i].first.x);
				int iy = static_cast<int>(py - corners[i].first.y);

				if (ix < 0 || iy < 0 || ix >= warpedMasks[i].size().width || iy >= warpedMasks[i].size().height) {
					continue;
				}

				if (warpedMasks[i].at<uchar>(iy, ix)) {
					nbValues++;
				}
			}

			if (nbValues == 0) {
				continue;
			}

			Mat values(nbValues, 3, CV_32F);
			vector<Vec3f> valuesVec(nbValues);
			float meanR = 0, meanG = 0, meanB = 0;
			nbValues = 0;

			for (int i = 0; i < _nbImages; ++i) {
				int ix = static_cast<int>(px - corners[i].first.x);
				int iy = static_cast<int>(py - corners[i].first.y);

				if (ix < 0 || iy < 0 || ix >= warpedMasks[i].size().width || iy >= warpedMasks[i].size().height) {
					continue;
				}

				if (warpedMasks[i].at<uchar>(iy, ix)) {
					float mask = static_cast<float>(warpedMasks[i].at<uchar>(iy, ix)) / 255;

					meanR += (values.at<float>(nbValues, 0) = (warpedImages[i].at<Vec3b>(iy, ix)[0] / mask));
					meanG += (values.at<float>(nbValues, 1) = (warpedImages[i].at<Vec3b>(iy, ix)[1] / mask));
					meanB += (values.at<float>(nbValues, 2) = (warpedImages[i].at<Vec3b>(iy, ix)[2] / mask));

					valuesVec[nbValues][0] = values.at<float>(nbValues, 0);
					valuesVec[nbValues][1] = values.at<float>(nbValues, 1);
					valuesVec[nbValues][2] = values.at<float>(nbValues, 2);

					nbValues++;
				}
			}

			meanR /= nbValues;
			meanG /= nbValues;
			meanB /= nbValues;

			sort(valuesVec.begin(), valuesVec.end(), compareVec3f);

			float stdDev = 0;
			Vec3f medianValue;
			int medianIndex = nbValues / 2;
			vector<float> diffVect;

			if (nbValues % 2 == 0) {
				medianValue = (valuesVec[medianIndex - 1] + valuesVec[medianIndex]) / 2;
			} else {
				medianValue = valuesVec[medianIndex];
			}

			for (int i = 0; i < nbValues; ++i) {
				float diff = std::max(std::max(std::abs(meanR - values.at<float>(i, 0)),
					std::abs(meanG - values.at<float>(i, 1))),
					std::abs(meanB - values.at<float>(i, 2)));

				stdDev += diff * diff;

				diff = std::max(std::max(std::abs(medianValue[0] - values.at<float>(i, 0)),
					std::abs(medianValue[1] - values.at<float>(i, 1))),
					std::abs(medianValue[2] - values.at<float>(i, 2)));

				diffVect.push_back(diff);
			}

			float mad;

			sort(diffVect.begin(), diffVect.end());
			stdDev = std::sqrtf(stdDev / nbValues);

			if (nbValues % 2 == 0) {
				mad = (diffVect[medianIndex - 1] + diffVect[medianIndex]) / 2;
			} else {
				mad = diffVect[medianIndex];
			}

			madImage(y, x) = mad;

			if (stdDev > minStdDev && nbValues >= nbClusters) {
				kmeans(values, nbClusters, labels, criteria, criteria.maxCount, KMEANS_RANDOM_CENTERS, centers);
				centers.convertTo(centers, CV_8U);

				int clusterMax = 0;

				for (int i = 0; i < nbClusters; ++i) {
					samplesCount[i] = 0;
				}

				for (int i = 0; i < nbValues; ++i) {
					if (++samplesCount[labels.at<int>(i, 0)] > samplesCount[clusterMax]) {
						clusterMax = labels.at<int>(i, 0);
					}
				}

				int sampleIndex = 0;

				valuesVec.resize(samplesCount[clusterMax]);
				diffVect.resize(samplesCount[clusterMax]);

				for (int i = 0; i < nbValues; ++i) {
					if (labels.at<int>(i, 0) != clusterMax) {
						continue;
					}

					valuesVec[sampleIndex][0] = values.at<float>(i, 0);
					valuesVec[sampleIndex][1] = values.at<float>(i, 1);
					valuesVec[sampleIndex++][2] = values.at<float>(i, 2);
				}

				sort(valuesVec.begin(), valuesVec.end(), compareVec3f);
				sampleIndex = 0;
				medianIndex = samplesCount[clusterMax] / 2;

				if (samplesCount[clusterMax] % 2 == 0) {
					medianValue = (valuesVec[medianIndex - 1] + valuesVec[medianIndex]) / 2;
				} else {
					medianValue = valuesVec[medianIndex];
				}

				for (int i = 0; i < nbValues; ++i) {
					if (labels.at<int>(i, 0) != clusterMax) {
						continue;
					}

					float diff = std::max(std::max(std::abs(medianValue[0] - values.at<float>(i, 0)),
						std::abs(medianValue[1] - values.at<float>(i, 1))),
						std::abs(medianValue[2] - values.at<float>(i, 2)));

					diffVect[sampleIndex++] = diff;
				}

				sort(diffVect.begin(), diffVect.end());

				if (samplesCount[clusterMax] % 2 == 0) {
					mad = (diffVect[medianIndex - 1] + diffVect[medianIndex]) / 2;
				} else {
					mad = diffVect[medianIndex];
				}

				madImage(y, x) = mad;

				finalImage.at<Vec3b>(y, x)[0] = saturate_cast<uchar>(centers.at<uchar>(clusterMax, 0));
				finalImage.at<Vec3b>(y, x)[1] = saturate_cast<uchar>(centers.at<uchar>(clusterMax, 1));
				finalImage.at<Vec3b>(y, x)[2] = saturate_cast<uchar>(centers.at<uchar>(clusterMax, 2));
			} else {
				finalImage.at<Vec3b>(y, x)[0] = saturate_cast<uchar>(meanR);
				finalImage.at<Vec3b>(y, x)[1] = saturate_cast<uchar>(meanG);
				finalImage.at<Vec3b>(y, x)[2] = saturate_cast<uchar>(meanB);
			}
		}
	}

	delete[] samplesCount;
	elapsedTime = static_cast<float>(clock() - start) / CLOCKS_PER_SEC;
	cout << endl << "  background total: " << elapsedTime << "s" << endl;

	start = clock();

	for (int interestImage = 0; interestImage < _nbImages; ++interestImage) {
		Mat baseImage = images.getImage(getImage(interestImage));
		const Size &size = baseImage.size();
		Mat_<Vec2f> unwarp(baseImage.size());
		Mat homography = getFullTransform(interestImage).clone();
		Mat translation = Mat::eye(Size(3, 3), CV_64F);

		translation.at<double>(0, 2) = -size.width / 2;
		translation.at<double>(1, 2) = -size.height / 2;
		homography = translation.inv() * homography * translation;
		translation.at<double>(0, 2) = finalMinCorner.x;
		translation.at<double>(1, 2) = finalMinCorner.y;
		homography = homography.inv() * translation;

		unwarp.setTo(Scalar(-1, -1));

		cout << "\r  Extracting foreground " << (interestImage + 1) << "/" << _nbImages << flush;

		Mat unwarpedBackground, difference, cleaned, unwarpedMad;
		Mat stdDev, mean;
		vector<Mat> channels(3);

		warpPerspective(finalImage, unwarpedBackground, homography, baseImage.size());
		warpPerspective(madImage, unwarpedMad, homography, baseImage.size());

		absdiff(unwarpedBackground, baseImage, difference);
		split(difference, channels);
		difference = max(channels[2], max(channels[1], channels[0]));

		Mat_<uchar> thresholded(size);

		thresholded.setTo(0);

		for (int y = 0; y < size.height; ++y) {
			uchar *thresholdedPtr = thresholded.ptr(y);
			float *madPtr = unwarpedMad.ptr<float>(y);
			uchar *differencePtr = difference.ptr<uchar>(y);

			for (int x = 0; x < size.width; ++x) {
				float mad = *madPtr++;
				float diff = *differencePtr;

				if (diff > mad * madForegroundThreshold * 1.4826) {
					*thresholdedPtr = 255;
				}

				*differencePtr = saturate_cast<uchar>(abs(diff / mad));

				differencePtr++;
				thresholdedPtr++;
			}
		}

		stringstream sstr;

		sstr << "output_foreground_" << setfill('0') << setw(4) << (interestImage + 1) << ".png";
		imwrite(sstr.str(), thresholded);

		int closingRadius = 2;
		Mat element = getStructuringElement(MORPH_RECT, Size(closingRadius * 2 + 1, closingRadius * 2 + 1), Point(closingRadius, closingRadius));
		Mat m0, m1;

		erode(thresholded, cleaned, element);

		element = getStructuringElement(MORPH_RECT, Size(3, 3), Point(1, 1));
		m1 = cleaned.clone();

		do {
			m0 = m1.clone();
			dilate(m0, m1, element);
			cv::min(m1, thresholded, m1);
		} while (countNonZero(abs(m0 - m1)) != 0);

		dilate(m1, cleaned, element);
		erode(cleaned, cleaned, element);

		sstr.str("");
		sstr << "output_foreground_clean_" << setfill('0') << setw(4) << (interestImage + 1) << ".png";
		imwrite(sstr.str(), cleaned);
	}
	
	elapsedTime = static_cast<float>(clock() - start) / CLOCKS_PER_SEC;
	cout << endl << "  foreground total: " << elapsedTime << "s" << endl;

	return finalImage;
}
