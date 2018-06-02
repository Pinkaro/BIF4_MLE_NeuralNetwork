using BIF4_MLE_UEB4.src;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BIF4_MLE_UEB4
{
    class Program
    {
        static void Main(string[] args)
        {
            double[] desiredValues = new double[10]
            {
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9
            };

            NeuralNetwork network = new NeuralNetwork(784, desiredValues, 0.05, 0.5, true, false);
            network.Train(0.01);
            network.Test();
            Console.ReadKey();
        }
    }
}
