#include "RegistrationOptimization.h"

#include <iostream>
#include <limits>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

namespace {
	const float eps = numeric_limits<float>::epsilon();

	Mat_<float> homographyToParameters(const Mat &homography)
	{
		Mat_<float> tau(8, 1);

		tau(0, 0) = homography.at<double>(0, 0) - 1;
		tau(1, 0) = homography.at<double>(1, 0);
		tau(2, 0) = homography.at<double>(2, 0);
		tau(3, 0) = homography.at<double>(0, 1);
		tau(4, 0) = homography.at<double>(1, 1) - 1;
		tau(5, 0) = homography.at<double>(2, 1);
		tau(6, 0) = homography.at<double>(0, 2);
		tau(7, 0) = homography.at<double>(1, 2);
		tau /= homography.at<double>(2, 2);

		return tau;
	}

	Mat_<double> parametersToHomography(const Mat_<float> &tau)
	{
		Mat_<double> homography(3, 3);

		homography(0, 0) = 1 + tau(0, 0);
		homography(1, 0) = tau(1, 0);
		homography(2, 0) = tau(2, 0);
		homography(0, 1) = tau(3, 0);
		homography(1, 1) = 1 + tau(4, 0);
		homography(2, 1) = tau(5, 0);
		homography(0, 2) = tau(6, 0);
		homography(1, 2) = tau(7, 0);
		homography(2, 2) = 1;

		return homography;
	}

	void registerImagesSingle(const Mat &sceneImage, const Mat &objectImage, Mat_<float> &tau, Mat_<float> &weight, Mat_<float> &residue)
	{
		int nbPixels = objectImage.size().width * objectImage.size().height;
		Mat_<float> X(nbPixels, 8);
		Mat dtau, y;
		Mat_<double> translation = Mat_<double>::eye(3, 3);
		Mat_<double> gradientKernel(1, 5);
		Mat_<float> mask;

		gradientKernel(0, 0) = 1.0 / 12;
		gradientKernel(0, 1) = -8.0 / 12;
		gradientKernel(0, 2) = 0;
		gradientKernel(0, 3) = 8.0 / 12;
		gradientKernel(0, 4) = -1.0 / 12;

		translation(0, 2) = -objectImage.size().width / 2;
		translation(1, 2) = -objectImage.size().height / 2;

		for (int iter = 0; iter < 50; ++iter) {
			Mat_<double> homography = parametersToHomography(tau);
			Mat objectImageWarped;

			//homography = translation.inv() * homography * translation;
			
			warpPerspective(objectImage, objectImageWarped, homography, sceneImage.size(), INTER_LINEAR, BORDER_REPLICATE);
			mask = Mat_<float>::ones(objectImage.size());
			warpPerspective(mask, mask, homography, sceneImage.size());

			Mat gradX, gradY;
			Mat img;

			filter2D(objectImageWarped, gradX, CV_32F, gradientKernel);
			filter2D(objectImageWarped, gradY, CV_32F, gradientKernel.t());

			y = sceneImage - objectImageWarped;
			
			cvtColor(y, y, CV_RGB2GRAY);
			multiply(y, mask, y);
			y = y.reshape(1, nbPixels);

			cvtColor(gradX, gradX, CV_RGB2GRAY);
			gradX = gradX.reshape(1, nbPixels);

			cvtColor(gradY, gradY, CV_RGB2GRAY);
			gradY = gradY.reshape(1, nbPixels);
		
			Mat_<float> xCoord(sceneImage.size()), yCoord(sceneImage.size());

			for (int y = 0; y < sceneImage.size().height; ++y) {
				for (int x = 0; x < sceneImage.size().width; ++x) {
					xCoord(y, x) = x;
					yCoord(y, x) = y;
				}
			}

			xCoord = xCoord.reshape(1, nbPixels);
			yCoord = yCoord.reshape(1, nbPixels);

			Mat_<float> D(nbPixels, 2);
			Mat column, column2;

			multiply(xCoord, gradX, column);
			column.copyTo(X.col(0));

			multiply(xCoord, gradY, column);
			column.copyTo(X.col(1));

			multiply(xCoord, xCoord, column);
			multiply(column, gradX, column);
			multiply(xCoord, yCoord, column2);
			multiply(column2, gradY, column2);
			column = column * -1 + column2 * -1;
			column.copyTo(X.col(2));

			multiply(yCoord, gradX, column);
			column.copyTo(X.col(3));

			multiply(yCoord, gradY, column);
			column.copyTo(X.col(4));

			multiply(yCoord, yCoord, column);
			multiply(column, gradY, column);
			multiply(xCoord, yCoord, column2);
			multiply(column2, gradX, column2);
			column = column * -1 + column2 * -1;
			column.copyTo(X.col(5));

			gradX.copyTo(X.col(6));
			gradY.copyTo(X.col(7));

			gradX.copyTo(D.col(0));
			gradY.copyTo(D.col(1));

			Mat_<float> tau2(2, 1);

			tau2(0, 0) = tau(2, 0);
			tau2(1, 0) = tau(5, 0);
			D = D * tau2 + 1 + eps;

			for (int i = 0; i < 8; ++i) {
				divide(X.col(i), D, X.col(i));
			}

			Mat A, wmul;

			weight = weight.reshape(1, nbPixels);
			A = X.clone();

			for (int i = 0; i < 8; ++i) {
				multiply(A.col(i), weight, A.col(i));
			}

			A = X.t() * A;
			A += Mat::diag(A.diag()) * 0.001;
			multiply(weight, y, wmul);

			weight = weight.reshape(1, objectImage.size().height);

			dtau = A.inv() * X.t() * wmul;
			tau += dtau;

			Mat dtau2;

			divide(dtau, tau + eps, dtau2);
			dtau2 = abs(dtau2);

			double minDtau, maxDtau;
			minMaxLoc(dtau2, &minDtau, &maxDtau);

			if (maxDtau < 0.001) {
				break;
			}
		}

		y -= X * dtau;
		residue = y.reshape(1, objectImage.size().height);
	}

	Mat_<float> influence(const Mat_<float> &x, float C)
	{
		Mat_<float> y = Mat_<float>::zeros(x.size());

		for (int i = 0; i < y.size().height; ++i) {
			const float *ptrx = x.ptr<float>(i);
			float *ptry = y.ptr<float>(i);

			for (int j = 0; j < y.size().width; ++j) {
				if (abs(*ptrx) < C) {
					*ptry = *ptrx * powf(C * C - *ptrx * *ptrx, 2);
				}

				ptrx++;
				ptry++;
			}
		}

		return y;
	}
}

Mat_<double> registerImages(const Mat &sceneImage, const Mat &objectImage, const Mat &initialHomography)
{
	Mat_<float> tau;
	Mat_<float> weight, residue;
	const int numLevels = 3;
	Mat_<double> translation = Mat_<double>::eye(3, 3), homography;
	double C;

	translation(0, 2) = -objectImage.size().width / 2;
	translation(1, 2) = -objectImage.size().height / 2;
	homography = translation.inv() * initialHomography * translation;
	tau = homographyToParameters(homography);
	
	for (int level = numLevels - 1; level >= numLevels - 1; --level) {
		Mat sceneImageResized, objectImageResized;
		double resizeFactor = pow(0.5, level);
		Size size = sceneImage.size();

		cout << "==== level " << level << endl;

		size.width *= resizeFactor;
		size.height *= resizeFactor;

		resize(sceneImage, sceneImageResized, size);
		resize(objectImage, objectImageResized, size);

		sceneImageResized.convertTo(sceneImageResized, CV_32FC3);
		objectImageResized.convertTo(objectImageResized, CV_32FC3);

		stringstream sstr;
		Mat warped, img;

		if (level == numLevels - 1) {
			weight = Mat_<float>::ones(size);
			tau(6, 0) *= resizeFactor * 2;
			tau(7, 0) *= resizeFactor * 2;
			
			double minValue, maxValue;
			Mat grayScale;

			cvtColor(sceneImage, grayScale, CV_RGB2GRAY);
			minMaxLoc(grayScale, &minValue, &maxValue);
			C = maxValue;
		} else {
			resize(weight, weight, size);
			tau(6, 0) *= 2;
			tau(7, 0) *= 2;
			C *= 2;
		}

		sstr << "pre" << level << "sce" << endl;
		cvNamedWindow(sstr.str().c_str(), CV_WINDOW_AUTOSIZE);
		sceneImageResized.convertTo(img, CV_8UC3);
		imshow(sstr.str().c_str(), img);

		sstr.str("");
		sstr << "pre" << level << "obj" << endl;
		cvNamedWindow(sstr.str().c_str(), CV_WINDOW_AUTOSIZE);
		warpPerspective(objectImageResized, warped, parametersToHomography(tau), sceneImageResized.size());
		warped.convertTo(img, CV_8UC3);
		imshow(sstr.str().c_str(), img);

		while(true) {
			for (int i = 0; i < 50; ++i) {
				cout << "iteration " << i << endl;
				Mat_<float> tauOld = tau.clone();
				double wmax, wmin;

				registerImagesSingle(sceneImageResized, objectImageResized, tau, weight, residue);
				divide(influence(abs(residue), C) + eps, abs(residue) + eps, weight);
				minMaxLoc(weight, &wmin, &wmax);
				weight /= wmax;

				Mat_<float> change;

				divide(tauOld - tau, tau + eps, change);
				minMaxLoc(abs(change), &wmin, &wmax);

				if (wmax < 0.01) {
					break;
				}
			}

			int nbPixels = size.width * size.height;
			int medianIndex = nbPixels / 2;
			float medianValue;

			residue = residue.reshape(1, nbPixels);
			cv::sort(residue, residue, CV_SORT_EVERY_COLUMN | CV_SORT_ASCENDING);

			if (nbPixels % 2 == 0) {
				medianValue = (residue(medianIndex - 1, 0) + residue(medianIndex, 0)) / 2;
			} else {
				medianValue = residue(medianIndex, 0);
			}

			residue -= medianValue;
			residue = abs(residue);
			cv::sort(residue, residue, CV_SORT_EVERY_COLUMN | CV_SORT_ASCENDING);

			if (nbPixels % 2 == 0) {
				medianValue = (residue(medianIndex - 1, 0) + residue(medianIndex, 0)) / 2;
			} else {
				medianValue = residue(medianIndex, 0);
			}

			float paraTurkey = 4.7 * 1.48 * medianValue;

			if (C >= max(paraTurkey, 0.0001f)) {
				C /= 2;
			} else {
				break;
			}
		}
		
		sstr.str("");
		sstr << "post" << level << "obj" << endl;
		cvNamedWindow(sstr.str().c_str(), CV_WINDOW_AUTOSIZE);
		warpPerspective(objectImageResized, warped, parametersToHomography(tau), sceneImageResized.size());
		warped.convertTo(img, CV_8UC3);
		imshow(sstr.str().c_str(), img);
	}

	homography = parametersToHomography(tau);
	homography = translation * homography * translation.inv();

	return homography;
}
