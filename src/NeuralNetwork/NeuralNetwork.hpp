#ifndef SRC_NEURALNETWORK_HPP
#define SRC_NEURALNETWORK_HPP

#include <vector>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include "../Neuron/Neuron.hpp"

class NeuralNetwork
{
    public:
        NeuralNetwork(const std::vector<unsigned> &topology);
        void feedForward(const std::vector<double> &inputVals);
        void backProp(const std::vector<double> &targetVals);
        void getResults(std::vector<double> &resultVals) const;
        double getRecentAverageError(void) const { return this->m_recentAverageError; }

    private:
        std::vector<Layer> m_layers; //m_layers[layerNum][neuronNum]
        double m_error;
        double m_recentAverageError;
        static double m_recentAverageSmoothingFactor;
};

#include "NeuralNetwork.ipp"

#endif