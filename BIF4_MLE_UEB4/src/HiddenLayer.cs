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

        internal double[][] weights;
        internal double[][] weightChanges;
        internal double[] biasWeights;
        internal double[] biasValues;

        public HiddenLayer(int hiddenNeuronsAmount)
        {
            NeuronValues = new double[hiddenNeuronsAmount];
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
            double sum = 0.0;

            for (int i = 0; i < Length; i++)
            {
                sum = 0.0;

                for (int j = 0; j < ChildLayer.Length; j++)
                {
                    sum += ChildLayer.Errors[j] * weights[i][j];
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
                    x += ParentLayer.NeuronValues[i] * ParentLayer.weights[i][j];
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
