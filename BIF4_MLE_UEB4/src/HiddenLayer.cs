using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BIF4_MLE_UEB4.src
{
    public class HiddenLayer : ALayer
    {
        public OutputLayer ChildLayer;
        public InputLayer ParentLayer;

        internal double[,] weights;
        internal double[,] weightChanges;
        internal double[] biasWeights;
        internal double[] biasValues;

        public HiddenLayer(int hiddenNeuronsAmount)
        {
            NeuronValues = new double[hiddenNeuronsAmount];
            InitializeWeights(NeuronValues.Length);
            Errors = new double[NeuronValues.Length];
        }

        private void InitializeWeights(int length)
        {
            if(weights == null && weightChanges == null && biasWeights == null && biasValues == null)
            {
                Random rand = new Random();

                weights = new double[length, length];
                RandomizeWeight(rand, null, weights);

                weightChanges = new double[length, length];
                RandomizeWeight(rand, null, weightChanges);

                biasWeights = new double[length];
                RandomizeWeight(rand, biasWeights, null);

                biasValues = new double[length];

                double currentBias = 0.0;
                for(int i = 0; i < biasValues.Length; i++)
                {
                    currentBias = rand.NextDouble() * 2.0 - 1.0;

                    if(currentBias >= 0)
                    {
                        biasValues[i] = 1.0;
                    }
                    else
                    {
                        biasValues[i] = -1.0;
                    }
                }
            }
            else
            {
                throw new InvalidOperationException("One or more array has already been assigned.");
            }
        }

        private void RandomizeWeight(Random rand, double[] single = null, double[,] multi = null)
        {
            if(single == null && multi != null)
            {
                for (int i = 0; i < multi.GetLength(0); i++)
                {
                    for (int j = 0; j < multi.GetLength(1); j++)
                    {
                        multi[i,j] = rand.NextDouble() * 2.0 - 1.0;
                    }
                }        
            }
            else if(single != null && multi == null)
            {
                for(int i = 0; i < single.Length; i++)
                {
                    single[i] = rand.NextDouble() * 2.0 - 1.0;
                }
            }
            else
            {
                throw new InvalidOperationException("One parameter has to be null.");
            }
        }

        public override void AdjustWeights()
        {
            double dw = 0.0;

            for (int i = 0; i < Length; i++)
            {
                for (int j = 0; j < ChildLayer.Length; j++)
                {
                    dw = NeuralNetwork.LearningRate * ChildLayer.Errors[j] * NeuronValues[i];

                    if (NeuralNetwork.UseMomentum)
                    {
                        weights[i,j] += dw + NeuralNetwork.MomentumFactor * weightChanges[i,j];

                        weightChanges[i,j] = dw;
                    }
                    else
                    {
                        weights[i,j] += dw;
                    }
                }
            }

            for (int j = 0; j < ChildLayer.Length; j++)
            {
                biasWeights[j] += NeuralNetwork.LearningRate * ChildLayer.Errors[j] * biasValues[j];
            }
        }

        public override void CalculateErrors()
        {
            double sum = 0.0;

            for (int i = 0; i < Length; i++)
            {
                sum = 0.0;

                for (int j = 0; j < ChildLayer.Length; j++)
                {
                    sum += ChildLayer.Errors[j] * weights[i,j];
                }

                Errors[i] = sum * NeuronValues[i] * (1.0 - NeuronValues[i]);
            }
        }

        public override void CalculateNeuronValues()
        {
            double x = 0.0;

            for (int j = 0; j < Length; j++)
            {
                x = 0.0;

                for (int i = 0; i < ParentLayer.Length; i++)
                {
                    x += ParentLayer.NeuronValues[i] * ParentLayer.weights[i,j];
                }

                x += ParentLayer.biasValues[j] * ParentLayer.biasWeights[j];

                if (NeuralNetwork.LinearOutput)
                {
                    NeuronValues[j] = x;
                }
                else
                {
                    NeuronValues[j] = 1.0 / (1.0 + Math.Exp(-x));
                }
            }
        }
    }
}
