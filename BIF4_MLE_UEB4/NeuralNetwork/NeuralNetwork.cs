using BIF4_MLE_NeuronalNetwork.utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BIF4_MLE_UEB4.NeuralNetwork
{
    public class NeuralNetwork
    {
        public static double LearningRate;
        public static double MomentumFactor;
        public static bool UseMomentum;
        public static bool LinearOutput;

        private InputLayer _inputLayer;
        private HiddenLayer _hiddenLayer;
        private OutputLayer _outputLayer;

        public NeuralNetwork(   int InputNeuronsAmount, double[] desiredValues, double learningRate, 
                                double momentumFactor, bool useMomentum, bool linearOutput)
        {
            InitializeLayers(InputNeuronsAmount, desiredValues);
            LearningRate = learningRate;
            MomentumFactor = momentumFactor;
            UseMomentum = useMomentum;
            LinearOutput = linearOutput;
        }

        private void InitializeLayers(int InputNeuronsAmount, double[] desiredValues)
        {
            _inputLayer = new InputLayer(InputNeuronsAmount);
            _outputLayer = new OutputLayer(desiredValues);

            int hiddenNeuronsAmount = (int)Math.Ceiling(Math.Sqrt(_inputLayer.Length * _outputLayer.Length));
            _hiddenLayer = new HiddenLayer(hiddenNeuronsAmount);

            // set parents and children
            _inputLayer.ChildLayer = _hiddenLayer;
            _outputLayer.ParentLayer = _hiddenLayer;
            _hiddenLayer.ChildLayer = _outputLayer;
            _hiddenLayer.ParentLayer = _inputLayer;
        }

        public void FeedForward()
        {
            _hiddenLayer.CalculateNeuronValues();
            _outputLayer.CalculateNeuronValues();
        }

        public double CalculateError()
        {
            double error = 0.0;

            for (int i = 0; i < _outputLayer.Length; i++)
            {
                error += Math.Pow(_outputLayer.NeuronValues[i] - _outputLayer.DesiredValues[i], 2);
            }

            error = error / _outputLayer.Length;
            return error;
        }

        public void BackPropagate()
        {
            _outputLayer.CalculateErrors();
            _hiddenLayer.CalculateErrors();
            _hiddenLayer.AdjustWeights();
            _inputLayer.AdjustWeights();
        }

        public void Train(double targetErrorRate = 0.05)
        {
            double totalError = 1.0;

            while(totalError > targetErrorRate)
            {
                foreach(Image image in MnistReader.ReadTrainingData())
                {
                    //_inputLayer.NeuronValues = image.Data;
                }
            }
        }
    }
}
