#include "ofxTSNE.h"


vector<vector<float>> ofxTSNE::run(vector<vector<float>>& data, int dims, double perplexity, double theta, bool runManually) {
	this->data = data;
	this->dims = dims;
	this->perplexity = perplexity;
	this->theta = theta;
	this->runManually = runManually;

	max_iter = 1000;
	iter = 0;

	N = data.size();
	D = data[0].size();

	if (N - 1 < 3 * perplexity) {
		ofLog(OF_LOG_WARNING, "Perplexity too large for number of data points, setting to max");
		perplexity = (float)(N - 1) / 3.0 - 1.0;
	}

	X = (double*)malloc(D * N * sizeof(double));
	Y = (double*)malloc(dims * N * sizeof(double));

	int idx = 0;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < D; j++) {
			X[idx] = data[i][j];
			idx++;
		}
	}

	// t-SNE
	tsne.run(X, N, D, Y, dims, perplexity, theta, runManually);

	if (runManually) {
		return tsnePoints;
	}
	return iterate();
}

vector<vector<float>> ofxTSNE::iterate() {
	if (iter > max_iter) {
		return tsnePoints;
	}

	if (runManually) {
		tsne.runIteration();
	}

	// unpack Y into tsnePoints
	tsnePoints.resize(N);
	int idxY = 0;
	for (int i = 0; i < N; i++) {
		auto& tsnePoint = tsnePoints[i];
		tsnePoint.resize(dims);
		for (int j = 0; j < dims; j++) {
			tsnePoint[j] = Y[idxY];
			idxY++;
		}
	}

	iter++;
	if (iter == max_iter) {
		finish();
	}

	return tsnePoints;
}

void ofxTSNE::finish() {
	delete(X);
	delete(Y);
}


vector<vector<float>> ofxTSNE::normalize(const vector<vector<float>>& data, ofxTSNENormalize normalize_mode)
{
	vector<vector<float>> result = data;
	if (normalize_mode == ofxTSNENormalize::None) {
		return result;
	}

	int N = data.size();
	int dims = data[0].size();

	vector<double> min_, max_;
	min_.resize(dims);
	max_.resize(dims);
	for (int i = 0; i < dims; i++) {
		min_[i] = numeric_limits<double>::max();
		max_[i] = numeric_limits<double>::min();
	}

	for (auto& point: result) {
		for (int j = 0; j < dims; j++) {
			if (point[j] < min_[j])  min_[j] = point[j];
			if (point[j] > max_[j])  max_[j] = point[j];
		}
	}

	if (normalize_mode == ofxTSNENormalize::Fit) {
		for (auto& point : result) {
			for (int j = 0; j < dims; j++) {
				point[j] = (point[j] - min_[j]) / (max_[j] - min_[j]);
			}
		}
	}
	if (normalize_mode == ofxTSNENormalize::RespectScale) {
		float L = 0.000001;
		for (int j = 0; j < dims; j++) {
			if (max_[j] - min_[j] > L) {
				L = max_[j] - min_[j];
			}
		}
		float Scale = 1.0f / L;
		for (auto& point : result) {
			for (int j = 0; j < dims; j++) {
				point[j] = (point[j] - (min_[j] + max_[j]) * 0.5) * Scale + 0.5;
			}
		}
	}
	return result;
}
