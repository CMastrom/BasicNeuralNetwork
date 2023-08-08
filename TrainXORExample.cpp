#include <iostream>
#include <vector>
#include <cstdint>
#include "src/NeuralNetwork/NeuralNetwork.hpp"

void coutVector(const std::string label, const std::vector<double> &v);

int main()
{
	const int trainingIterations = 4000;
	NeuralNetwork myNet({ 2, 4, 1 });

	std::vector<std::vector<double>> inputVals = {
		{
			0.0, 0.0
		},
		{
			1.0, 0.0
		},
		{
			0.0, 1.0
		},
		{
			1.0, 1.0
		}
	};

	std::vector<std::vector<double>> targetVals = {
		{
			0.0
		},
		{
			1.0
		},
		{
			1.0
		},
		{
			0.0
		}
	};

	std::vector<double> resultVals;
	uint8_t valueIndex = 0;

	for (unsigned trainingIteration = 1; trainingIteration <= trainingIterations; trainingIteration++)
	{
		std::cout << std::endl << "Training Iteration: " << trainingIteration << std::endl;

		coutVector("Input", inputVals[valueIndex]);
		myNet.feedForward(inputVals[valueIndex]);

		// Collect the net's actual results:
		myNet.getResults(resultVals);
		coutVector("Output", resultVals);

		// Train the net what the outputs should have been:
		coutVector("Target", targetVals[valueIndex]);
		myNet.backProp(targetVals[valueIndex]);

		if (
			++valueIndex > 3
		)
			valueIndex = 0;

		std::cout << "Net recent average error: " << myNet.getRecentAverageError() << std::endl;
	}

	std::cout << std::endl << "Done" << std::endl;

}

void coutVector(const std::string label, const std::vector<double> &v)
{
	std::cout << label << ": { ";
	for(unsigned i = 0; i < v.size(); ++i)
	{
		std::cout << v[i] << " ";
	}
	std::cout << "}" << std::endl;
}