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
        private HashSet<double> _errors;
        private int[,] _confusionMatrix;

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
            _errors = new HashSet<double>();

            InitializeConfusionMatrix();
        }

        private void InitializeConfusionMatrix()
        {
            _confusionMatrix = new int[10, 10];

            for (int i = 0; i < _confusionMatrix.GetLength(0); i++)
            {
                for (int j = 0; j < _confusionMatrix.GetLength(1); j++)
                {
                    _confusionMatrix[i, j] = 0;
                }
            }
        }

        private void PrintConfusionMatrix()
        {
            int distance = 5;
            string header = String.Format(
                "   {0," + distance + "}" + "{1," + distance + "}" + "{2," + distance + "}" + "{3," + distance + "}" +
                "{4," + distance + "}" + "{5," + distance + "}" + "{6," + distance + "}" + "{7," + distance + "}" +
                "{8," + distance + "}" + "{9," + distance + "}", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9);

            Console.WriteLine(header);

            for (int x = 0; x < header.Length + 2; x++)
            {
                Console.Write("-");
            }

            Console.WriteLine();
            for (int i = 0; i < _confusionMatrix.GetLength(0); i++)
            {
                Console.Write(i + " | ");
                for (int j = 0; j < _confusionMatrix.GetLength(1); j++)
                {
                    Console.Write(String.Format("{0," + distance + "}", _confusionMatrix[i, j]));
                }

                Console.WriteLine();
            }
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
                    newDesiredValues[i] = 1.0;
                }
                else
                {
                    newDesiredValues[i] = 0.0;
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

            for (int i = 0; i < _outputLayer.Length; i++)
            {
                error += Math.Pow(_outputLayer.NeuronValues[i] - _outputLayer.DesiredValues[i], 2);
            }

            error = error / _outputLayer.Length;
            return error;
        }

        public double CalculateTotalError()
        {
            double overallError = 0.0;

            foreach (double singleError in _errors)
            {
                overallError += singleError;
            }

            return overallError / (double)_errors.Count();
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
            int count = 0;
            int sum = 0;
            int epochs = 0;
            double totalError = 1.0;

            while (totalError > targetErrorRate)
            {
                epochs++;
                _errors.Clear();
                foreach (Image image in MnistReader.ReadTrainingData())
                {

                    count++;
                    _inputLayer.SetNeurons(image.Data);
                    SetNewDesiredValues(image.Label);
                    FeedForward();
                    _errors.Add(CalculateError());
                    BackPropagate();


                    if (count == 1000)
                    {
                        sum += count;
                        count = 0;

                        Console.WriteLine("Images read: " + sum);
                    }
                }
                

                totalError = CalculateTotalError();
                Console.WriteLine("Current total error: " + totalError + "\tEpoch " + epochs);
            }

            Console.WriteLine("Finished training with total error of: " + totalError);
        }

        public void Test()
        {
            double testDataAmount = 0;
            double correctEstimate = 0;
            foreach (Image image in MnistReader.ReadTestData())
            {
                testDataAmount++;
                _inputLayer.SetNeurons(image.Data);
                SetNewDesiredValues(image.Label);

                FeedForward();
                CalculateError();

                _confusionMatrix[(int) image.Label, _outputLayer.GetIndexOfHighestNeuron()]++;

                for (int i = 0; i < _outputLayer.DesiredValues.Length; i++)
                {
                    if (_outputLayer.DesiredValues[i] == 1.0)
                    {
                        if (i == _outputLayer.GetIndexOfHighestNeuron())
                        {
                            correctEstimate++;
                        }
                    }
                }
            }

            double accuracy = (correctEstimate / testDataAmount) * 100;
            Console.WriteLine("Accuracy: " + accuracy + " %.\n");
            PrintConfusionMatrix();
        }
    }
}
