#ifndef SRC_NEURON_NEURON_HPP
#define SRC_NEURON_NEURON_HPP

#include <vector>
#include <cmath>

/**
 * A neural layer is comprised of an 
 * array of neurons.
*/
class Neuron;
typedef std::vector<Neuron> Layer;

/**
 * Outlines a basic
 * connection between
 * neurons.
*/
struct Connection
{
	double weight;
	double deltaWeight;
};

class Neuron
{
    public:
        Neuron(unsigned numOutputs, unsigned myIndex);
        void setOutputVal(double val) { this->m_outputVal = val; }
        double getOutputVal(void) const { return this->m_outputVal; }
        void feedForward(const Layer &prevLayer);
        void calcOutputGradients(double targetVals);
        void calcHiddenGradients(const Layer &nextLayer);
        void updateInputWeights(Layer &prevLayer);

    private:
        static double eta;
        static double alpha;
        static double transferFunction(double x);
        static double transferFunctionDerivative(double x);
        static double randomWeight(void) { return rand() / double(RAND_MAX); }
        double sumDOW(const Layer &nextLayer) const;
        double m_outputVal;
        std::vector<Connection> m_outputWeights;
        unsigned m_myIndex;
        double m_gradient;
};

#include "Neuron.ipp"

#endif