using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BIF4_MLE_UEB4.src
{
    public class InputLayer : ALayer
    {
        internal double[,] weights;
        internal double[,] weightChanges;
        internal double[] biasWeights;
        internal double[] biasValues;

        public HiddenLayer ChildLayer;

        public InputLayer(int neuronAmount)
        {
            NeuronValues = new double[neuronAmount];
            InitializeWeights(NeuronValues.Length);
        }

        private void InitializeWeights(int length)
        {
            if (weights == null && weightChanges == null && biasWeights == null && biasValues == null)
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
                for (int i = 0; i < biasValues.Length; i++)
                {
                    currentBias = rand.NextDouble() * 2.0 - 1.0;

                    if (currentBias >= 0)
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
            if (single == null && multi != null)
            {
                for (int i = 0; i < multi.GetLength(0); i++)
                {
                    for (int j = 0; j < multi.GetLength(1); j++)
                    {
                        multi[i, j] = rand.NextDouble() * 2.0 - 1.0;
                    }
                }
            }
            else if (single != null && multi == null)
            {
                for (int i = 0; i < single.Length; i++)
                {
                    single[i] = rand.NextDouble() * 2.0 - 1.0;
                }
            }
            else
            {
                throw new InvalidOperationException("One parameter has to be null.");
            }
        }

        public void SetNeurons(double[,] input)
        {
            this.NeuronValues = new double[input.Length];

            for(int i = 0; i < input.GetLength(0); i++)
            {
                for(int j = 0; j < input.GetLength(1); j++)
                {
                    this.NeuronValues[j] = input[i, j];
                }
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
            throw new InvalidOperationException("This operation is not allowed on InputLayer.");
        }

        public override void CalculateNeuronValues()
        {
            throw new InvalidOperationException("This operation is not allowed on InputLayer.");
        }
    }
}
