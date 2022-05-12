#include "Neural.h"
#include <stdlib.h>
#include <time.h>
#include <fstream>

NeuralNetwork::NeuralNetwork(int inputLayer, std::vector<int> hiddenLayer, int outputLayer, double learningrate)
{
	srand((unsigned int)time(0));
	this->inputLayer = inputLayer;
	this->hiddenLayer = hiddenLayer;
	this->outputLayer = outputLayer;
	this->learningrate = learningrate;
	neuronLayers.push_back(new RowVector(inputLayer));
	cacheLayers.push_back(new RowVector(inputLayer));
	delta.push_back(new RowVector(inputLayer));
	for (int i = 0; i < hiddenLayer.size(); ++i)
	{
		neuronLayers.push_back(new RowVector(hiddenLayer[i]));
		cacheLayers.push_back(new RowVector(hiddenLayer[i]));
		delta.push_back(new RowVector(hiddenLayer[i]));
	}
	neuronLayers.push_back(new RowVector(outputLayer));
	cacheLayers.push_back(new RowVector(outputLayer));
	delta.push_back(new RowVector(outputLayer));
	weights.push_back(new Matrix(inputLayer, hiddenLayer[0]));
	deltasWeight.push_back(new Matrix(inputLayer, hiddenLayer[0]));
	*weights.back() = 0 + (Eigen::ArrayXXd::Random(inputLayer, hiddenLayer[0]) * 0.5 + 0.5) * (1);
	*deltasWeight.back() = Eigen::ArrayXXd::Zero(inputLayer, hiddenLayer[0]);

	for (int i = 0; i < hiddenLayer.size() - 1; ++i)
	{
		weights.push_back(new Matrix(hiddenLayer[i], hiddenLayer[i + 1]));
		deltasWeight.push_back(new Matrix(hiddenLayer[i], hiddenLayer[i + 1]));
		*weights.back() = 0 + (Eigen::ArrayXXd::Random(hiddenLayer[i], hiddenLayer[i + 1]) * 0.5 + 0.5) * (1);
		*deltasWeight.back() = Eigen::ArrayXXd::Zero(hiddenLayer[i], hiddenLayer[i + 1]);
	}
	weights.push_back(new Matrix(hiddenLayer[hiddenLayer.size() - 1], outputLayer));
	deltasWeight.push_back(new Matrix(hiddenLayer[hiddenLayer.size() - 1], outputLayer));
	*weights.back() = 0 + (Eigen::ArrayXXd::Random(hiddenLayer[hiddenLayer.size() - 1], outputLayer) * 0.5 + 0.5) * (1);
	*deltasWeight.back() = Eigen::ArrayXXd::Zero(hiddenLayer[hiddenLayer.size() - 1], outputLayer);
	input_training_sets();
}

void NeuralNetwork::forward_pass()
{

	for (int i = 1; i < neuronLayers.size(); i++)
	{

		*neuronLayers[i] = (*neuronLayers[i - 1]) * (*weights[i - 1]);
		*cacheLayers[i] = *neuronLayers[i];
	}

	for (int i = 1; i < neuronLayers.size(); i++)
	{
		RowVector a = *neuronLayers[i];
		for (int j = 0; j < a.size(); j++)
		{
			a(j) = activation_function(a(j));
		}
		*neuronLayers[i] = a;
	}
}

void NeuralNetwork::calcErrors(RowVector output)
{
	for (int i = 0; i < delta.back()->cols(); ++i)
	{
		delta.back()->coeffRef(0, i) = -(output.coeffRef(0, i) - neuronLayers.back()->coeffRef(0, i)) * neuronLayers.back()->coeffRef(0, i) * (1 - neuronLayers.back()->coeffRef(0, i));
	}
	for (int i = delta.size() - 2; i > 0; --i)
	{
		for (int j = 0; j < delta[i]->cols(); ++j)
		{
			delta[i]->coeffRef(0, j) = calcDeltasHidden(i, j) * neuronLayers[i]->coeffRef(0, j) * (1 - neuronLayers[i]->coeffRef(0, j));
		}
	}
	updateWeights();
}

double NeuralNetwork::calcDeltasHidden(int x, int y)
{
	double result = 0;
	for (int i = 0; i < delta[x + 1]->cols(); ++i)
	{
		result = delta[x + 1]->coeffRef(0, i) * weights[x]->coeffRef(y, i);
	}
	return result;
}

void NeuralNetwork::updateWeights()
{
	for (int i = weights.size() - 1; i >= 0; --i)
	{
		for (int x = 0; x < weights[i]->rows(); ++x)
		{
			for (int y = 0; y < weights[i]->cols(); ++y)
			{
				double result = weights[i]->coeffRef(x, y) - (learningrate * neuronLayers[i]->coeffRef(0, x) * delta[i + 1]->coeffRef(0, y));
				weights[i]->coeffRef(x, y) = weights[i]->coeffRef(x, y) - (learningrate * neuronLayers[i]->coeffRef(0, x) * delta[i + 1]->coeffRef(0, y));
			}
		}
	}
}

double NeuralNetwork::activation_function(double x)
{
	return 1 / (1 + (exp(-x)));
}

void NeuralNetwork::read_input(RowVector input)
{
	*neuronLayers.front() = input;
	*cacheLayers.front() = input;
}

void NeuralNetwork::print_network()
{
	for (int i = 0; i < neuronLayers.size(); ++i)
	{
		std::cout << *neuronLayers[i] << "\n\n";
	}
}

void NeuralNetwork::print_weights()
{
	for (int i = 0; i < weights.size(); ++i)
	{
		std::cout << *weights[i] << "\n\n";
	}
}

void NeuralNetwork::print_deltas()
{
	for (int i = 0; i < deltasWeight.size(); ++i)
	{
		std::cout << *deltasWeight[i] << "\n\n";
	}
}

void NeuralNetwork::input_training_sets()
{
	std::ifstream f("input.csv");
	trainingsets = 0;
	while (!f.eof())
	{
		inputTraining.push_back(new RowVector(inputLayer));
		outputTraining.push_back(new RowVector(outputLayer));
		for (int i = 0; i < inputLayer; ++i)
		{
			f >> inputTraining[trainingsets]->coeffRef(0, i);
		}
		for (int i = 0; i < outputLayer; ++i)
		{
			f >> outputTraining[trainingsets]->coeffRef(0, i);
		}

		trainingsets++;
	}
	f.close();
	output_training_sets();
}

void NeuralNetwork::output_training_sets()
{
	for (int i = 0; i < trainingsets; ++i)
	{
		double szam1 = inputTraining[i]->coeffRef(0, 0);
		std::cout << *inputTraining[i] << "////" << *outputTraining[i] << '\n';
	}
}

void NeuralNetwork::training_nn()
{
	int epoch = 25000;
	for (int i = 0; i < epoch; ++i)
	{
		int trainingnumber = rand() % (trainingsets);
		read_input(*inputTraining[trainingnumber]);
		forward_pass();
		calcErrors(*outputTraining[trainingnumber]);
		if (i == epoch / 2)
		{
			learningrate = learningrate * 0.5;
		}
		calculateError(*outputTraining[trainingnumber]);
	}
}

void NeuralNetwork::makeprediction(RowVector input)
{
	read_input(input);
	forward_pass();
	std::cout << *neuronLayers.back() << '\n';
}

double NeuralNetwork::calculateError(RowVector output)
{
	double result = 0;
	for (int i = 0; i < outputLayer; ++i)
	{
		result += ((output.coeffRef(0, i) - neuronLayers.back()->coeffRef(0, i))) * ((output.coeffRef(0, i) - neuronLayers.back()->coeffRef(0, i)));
	}
	std::cout << (0.5 * result) << '\n';
	return ((1 / 2) * result);
}

void NeuralNetwork::test_network()
{
	std::ifstream f("test.csv");
	inputTest.push_back(new RowVector(inputLayer));
	outputTest.push_back(new RowVector(outputLayer));
	while (!f.eof())
	{

		for (int i = 0; i < inputLayer; ++i)
		{
			f >> inputTraining[0]->coeffRef(0, i);
		}
		for (int i = 0; i < outputLayer; ++i)
		{
			f >> outputTraining[0]->coeffRef(0, i);
		}
		std::cout << *outputTraining[0] << '\n';
		makeprediction(*inputTraining[0]);
		std::cout << "////////////" << '\n';
	}
	f.close();
}
