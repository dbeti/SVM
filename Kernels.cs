using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SVM
{
    /// <summary>
    /// Sučelje za jezgrine funkcije.
    /// </summary>
    public interface IKernel
    {
        /// <summary>
        /// U metodi je potrebno implementirati funkciju. 
        /// </summary>
        double Compute(double[] x1, double[] x2);

        /// <summary>
        /// Ime jezgre.
        /// </summary>
        string Name { get; set; }
    }

    /// <summary>
    /// Polinomna jezgra sa odmakom <see cref="Offset"/> i stupnjem <see cref="Dimension"/>. 
    /// </summary>
    public class PolynomialKernel : IKernel
    {
        private string name;
        public double Dimension { get; private set; }

        public double Offset { get; private set; }

        public string Name
        {
            get { return name; }
            set { name = value; }
        } 

        public PolynomialKernel(params double[] arguments)
        {
            Name = $"polynomial-{arguments[0]}-{arguments[1]}";
            Dimension = arguments[0];
            Offset = arguments[1];
        }

        public PolynomialKernel(double dimension, double offset)
        {
            Name = $"polynomial-{dimension}-{offset}";
            Dimension = dimension;
            Offset = offset;
        }

        public double Compute(double[] x1, double[] x2)
        {
            if (x1.Length != x2.Length)
                throw new ArgumentException("Arrays don't have same length!");

            double dot = 0;
            for (int i = 0; i < x1.Length; ++i)
                dot += x1[i] * x2[i];

            return Math.Pow(Offset + dot, Dimension);
        }
    }

    /// <summary>
    /// Gaussova jezgra s parametrom <see cref="Sigma"/>. 
    /// </summary>
    public class GaussKernel : IKernel
    {
        private string name;

        public double Sigma { get; private set; }

        public string Name
        {
            get { return name; }
            set { name = value; }
        }

        public GaussKernel(double sigma)
        {
            Name = $"gauss-{sigma}";
            Sigma = sigma;
        }

        public double Compute(double[] x1, double[] x2)
        {
            double norm = 0, tmp = 0;

            for (int i = 0; i < x1.Length; ++i)
            {
                tmp = x1[i] - x2[i];
                tmp = tmp * tmp;
                norm += tmp;
            }

            norm = (-1) * norm / (2 * Sigma * Sigma);
            norm = Math.Exp(norm);

            return norm;
        }
    }
}
