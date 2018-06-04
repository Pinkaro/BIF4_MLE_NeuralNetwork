using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BIF4_MLE_UEB4.src
{
    public class OutputLayer : ALayer
    {
        public double[] DesiredValues;
        public HiddenLayer ParentLayer;

        public OutputLayer(int desiredValuesLength)
        {
            NeuronValues = new double[desiredValuesLength];
            Errors = new double[NeuronValues.Length];
        }

        public override void AdjustWeights()
        {
            throw new InvalidOperationException("This operation is not allowed on OutputLayer.");
        }

        public override void CalculateErrors()
        {
            for (int i = 0; i < Length; i++)
            {
                Errors[i] = (DesiredValues[i] - NeuronValues[i]) * NeuronValues[i] * (1.0 - NeuronValues[i]);
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
                    x += ParentLayer.NeuronValues[i] * ParentLayer.weights[i, j];
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

        public int GetIndexOfHighestNeuron()
        {
            int highestNeuronIndex = 0;
            double highestNeuron = 0.0;

            for (int i = 0; i < NeuronValues.Length; i++)
            {
                if (NeuronValues[i] > highestNeuron)
                {
                    highestNeuron = NeuronValues[i];
                    highestNeuronIndex = i;
                }
            }

            return highestNeuronIndex;
        }
    }
}
