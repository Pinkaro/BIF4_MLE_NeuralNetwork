using BIF4_MLE_NeuronalNetwork.utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BIF4_MLE_UEB4.src
{
    public class NeuralNetwork
    {
        public static double LearningRate;
        public static double MomentumFactor;
        public static bool UseMomentum;
        public static bool LinearOutput;
        public static int InputAmount;

        private InputLayer _inputLayer;
        private HiddenLayer _hiddenLayer;
        private OutputLayer _outputLayer;
        private double[] _desiredValueBlueprint;
        private int _currentGuess;

        public NeuralNetwork(int InputNeuronsAmount, double[] desiredValues, double learningRate,
                                double momentumFactor, bool useMomentum, bool linearOutput)
        {
            _desiredValueBlueprint = desiredValues;
            InputAmount = InputNeuronsAmount;
            InitializeLayers(InputNeuronsAmount);
            LearningRate = learningRate;
            MomentumFactor = momentumFactor;
            UseMomentum = useMomentum;
            LinearOutput = linearOutput;
        }

        private void InitializeLayers(int InputNeuronsAmount)
        {
            _inputLayer = new InputLayer(InputNeuronsAmount);
            _outputLayer = new OutputLayer(_desiredValueBlueprint.Length);

            int hiddenNeuronsAmount = (int)Math.Ceiling(Math.Sqrt(_inputLayer.Length * _outputLayer.Length));
            _hiddenLayer = new HiddenLayer(hiddenNeuronsAmount);

            // set parents and children
            _inputLayer.ChildLayer = _hiddenLayer;
            _outputLayer.ParentLayer = _hiddenLayer;
            _hiddenLayer.ChildLayer = _outputLayer;
            _hiddenLayer.ParentLayer = _inputLayer;
        }

        private void SetNewDesiredValues(double target)
        {
            double[] newDesiredValues = new double[_desiredValueBlueprint.Length];

            for (int i = 0; i < newDesiredValues.Length; i++)
            {
                if (_desiredValueBlueprint[i] == target)
                {
                    newDesiredValues[i] = 1;
                }
                else
                {
                    newDesiredValues[i] = 0;
                }
            }

            _outputLayer.DesiredValues = newDesiredValues;
        }

        public void FeedForward()
        {
            _hiddenLayer.CalculateNeuronValues();
            _outputLayer.CalculateNeuronValues();
        }

        public double CalculateError()
        {
            double error = 0.0;
            double highestEstimate = 0.0;

            for (int i = 0; i < _outputLayer.Length; i++)
            {
                error += Math.Pow(_outputLayer.NeuronValues[i] - _outputLayer.DesiredValues[i], 2);

                if(_outputLayer.NeuronValues[i] > highestEstimate)
                {
                    highestEstimate = _outputLayer.NeuronValues[i];
                    _currentGuess = i;
                }
                
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

            foreach (Image image in MnistReader.ReadTrainingData())
            {
                _inputLayer.SetNeurons(image.Data);
                SetNewDesiredValues(image.Label);
                FeedForward();
                BackPropagate();
                totalError = CalculateError();

                if (totalError <= targetErrorRate)
                {
                    Console.WriteLine("\nAccurate set found: " + totalError + "(" + _currentGuess + "/" + image.Label + ")");
                    break;
                }

                if(_currentGuess == image.Label)
                {
                    Console.WriteLine("Current total error: " + totalError + " // Estimate correct (" + _currentGuess + "/" + image.Label +")");
                }
                else
                {
                    Console.WriteLine("Current total error: " + totalError + " // Estimate false (" + _currentGuess + "/"+ image.Label +")");
                }

                
            }

        }

        public void Test()
        {
            double count = 0;
            double accurate = 0;
            foreach (Image image in MnistReader.ReadTestData())
            {
                count++;
                _inputLayer.SetNeurons(image.Data);
                SetNewDesiredValues(image.Label);

                FeedForward();
                CalculateError();

                if (_currentGuess == image.Label)
                {
                    accurate++;
                }
            }

            double accuracy = (accurate / count) * 100;
            Console.WriteLine("Accuracy: " + accuracy + " %");
        }
    }
}
