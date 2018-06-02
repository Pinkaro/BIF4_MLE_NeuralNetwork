using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BIF4_MLE_UEB4.NeuralNetwork
{
    public abstract class ALayer
    {
        private double[] _neuronValues;
        public double[] NeuronValues
        {
            get { return _neuronValues; }
            set { _neuronValues = value; }
        }

        public int Length
        {
            get { return _neuronValues.Length; }
        }

        public double[] Errors;


        public abstract void CalculateErrors();

        public abstract void AdjustWeights();

        public abstract void CalculateNeuronValues();
    }
}
