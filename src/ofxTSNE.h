#pragma once

#include "ofMain.h"
#include "tsne.h"

enum class ofxTSNENormalize : int {
	None = 0,           // No normalizing
	Fit = 1,            // Normalize to [0,1] without respecting scale
	RespectScale = 2    // Normalize respecting scale
};

class ofxTSNE
{
public:
	vector<vector<float>> run(vector<vector<float>>& data, int dims = 2, double perplexity = 30, double theta = 0.5, bool runManually = false);
	vector<vector<float>> iterate();

	// Useful for normalizing for visualizing and keeping original embedding scale
	static vector<vector<float>> normalize(const vector<vector<float>>& data, ofxTSNENormalize normalize_mode);
private:
	void finish();

	TSNE tsne;
	vector<vector<float>> tsnePoints;

	vector<vector<float>> data;
	int dims;
	double perplexity;
	double theta;
	bool runManually;

	int N, D;
	double* X, * Y;

	int iter, max_iter;
};
