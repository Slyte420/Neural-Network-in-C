#include <iostream>
#include "Neural.h"
#include <vector>
int main() {
	int inputLayer = 2;
	std::vector<int> hiddenLayer;
	hiddenLayer.push_back(5);
	hiddenLayer.push_back(5);
	int outputLayer = 5;
	double learningrate = 0.01;

	NeuralNetwork net(inputLayer, hiddenLayer, outputLayer, learningrate);

	net.training_nn();
	net.test_network();
	
	return 0;
}