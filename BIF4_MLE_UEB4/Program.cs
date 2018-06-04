using BIF4_MLE_UEB4.src;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
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

            NeuralNetwork network = new NeuralNetwork(784, desiredValues, 0.2, 0.9, true, false);
            //network.Train(0.005);
            //network.Test();

            int[,] testMatrix = new int[10, 10];
            Random rand = new Random();

            for (int i = 0; i < testMatrix.GetLength(0); i++)
            {
                for (int j = 0; j < testMatrix.GetLength(1); j++)
                {
                    testMatrix[i, j] = rand.Next(0, 1000);
                }
            }

            


            Console.ReadKey();
        }
    }
}
