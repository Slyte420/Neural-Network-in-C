#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <math.h>
typedef Eigen::MatrixXd Matrix;
typedef Eigen::RowVectorXd RowVector;
typedef Eigen::VectorXd ColVector;

class NeuralNetwork
{
public:
	NeuralNetwork(int inputLayer, std::vector<int> hiddenLayer, int outputLayer, double learningrate);
	void forward_pass();
	double activation_function(double x);
	void read_input(RowVector input);
	void calcErrors(RowVector output);
	double calcDeltasHidden(int x, int y);
	void updateWeights();
	void print_network();
	void print_weights();
	void print_deltas();
	void input_training_sets();
	void output_training_sets();
	void training_nn();
	void makeprediction(RowVector input);
	double calculateError(RowVector output);
	void test_network();

	int inputLayer;
	std::vector<int> hiddenLayer;
	int outputLayer;
	double learningrate;
	int trainingsets;
	std::vector<RowVector *> neuronLayers;
	std::vector<RowVector *> cacheLayers;
	std::vector<RowVector *> delta;
	std::vector<Matrix *> deltasWeight;
	std::vector<Matrix *> weights;
	std::vector<RowVector *> inputTraining;
	std::vector<RowVector *> outputTraining;
	std::vector<RowVector *> inputTest;
	std::vector<RowVector *> outputTest;
};
