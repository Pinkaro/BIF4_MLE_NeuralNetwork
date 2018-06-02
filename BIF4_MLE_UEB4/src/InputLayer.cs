using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BIF4_MLE_UEB4.src
{
    public class InputLayer : ALayer
    {
        internal double[][] weights;
        internal double[][] weightChanges;
        internal double[] biasWeights;
        internal double[] biasValues;

        public HiddenLayer ChildLayer;

        public InputLayer(int neuronAmount)
        {
            NeuronValues = new double[neuronAmount];
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
                        weights[i][j] += dw + NeuralNetwork.MomentumFactor * weightChanges[i][j];

                        weightChanges[i][j] = dw;
                    }
                    else
                    {
                        weights[i][j] += dw;
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
