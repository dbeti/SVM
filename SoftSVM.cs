using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SVM
{
    /// <summary>
    /// Implementacija SVM-a s mekom marginom i parametrom Nu.
    /// </summary>
    public class SoftSVM : SVM
    {
        public SoftSVM(string name, double[][] values, double[] labels, IKernel kernel, string category) : base(name, values, labels, kernel, category)
        {
        }

        public double Nu { get; set; } = 0.08;

        public double B { get; set; }

        public double Lambda { get; set; }

        public double Margin { get; set; }

        public override string AlgorithmName => "Soft";

        public override void Algorithm()
        {
            SupportVectors.Clear();
            SupportVectors.Capacity = 0;
            // Save support vectors indexes
            for (int i = 0; i < Solver.lagrangians.Length; ++i)
                if (Solver.lagrangians[i] > 0)
                    SupportVectors.Add(i);

            int indexi = -1, indexj = -1;
            foreach (var i in SupportVectors)
            {
                if ((-1.0) / (Nu * Values.Length) < Solver.lagrangians[i] * Labels[i] && Solver.lagrangians[i] * Labels[i] < 0)
                    indexi = i;
                if (0 < Solver.lagrangians[i] * Labels[i] && Solver.lagrangians[i] * Labels[i] < 1.0 / (Nu * Values.Length))
                    indexj = i;

                foreach (var j in SupportVectors)
                {
                    Lambda += Labels[i] * Labels[j] * Solver.lagrangians[i] * Solver.lagrangians[j] * KernelMatrix[i, j];
                }
            }

            Lambda = Math.Sqrt(Lambda) / 2.0;

            double left = 0, right = 0;
            foreach (var k in SupportVectors)
            {
                left += Solver.lagrangians[k] * Labels[k] * KernelMatrix[k, indexi];
                right += Solver.lagrangians[k] * Labels[k] * KernelMatrix[k, indexj];
            }

            B = -Lambda * (left + right);
            Margin = 2 * Lambda * right + B;
        }

        public override double Decide(double[] input)
        {
            double result = 0;
            foreach (int index in SupportVectors)
                result += Solver.lagrangians[index] * Labels[index] * Kernel.Compute(Values[index], input);

            result += B;

            return Math.Sign(result);
        }

        public override void SetConstraints()
        {
            if (Solver.lbnd == null || Solver.lbnd.Length != Values.Length)
            {
                Solver.lbnd = new double[Values.Length];
                Solver.ubnd = new double[Values.Length];
                Solver.c = new double[2, Values.Length + 1];
                Solver.ct = new int[] { 0, 0 };
                Solver.s = new double[Values.Length];
            }

            for (int i = 0; i < Values.Length; ++i)
            {
                Solver.lbnd[i] = 0;
                Solver.ubnd[i] = 1.0 / (Nu * Values.Length);
                Solver.c[0, i] = Labels[i];
                Solver.c[1, i] = 1;
                Solver.s[i] = 1;
            }

            Solver.c[0, Values.Length] = 0;
            Solver.c[1, Values.Length] = 1;
        }

        public override void SetLinearTerm()
        {
            if (Solver.l == null || Solver.l.Length != Values.Length)
                Solver.l = new double[Values.Length];

            Solver.l.Populate(0);
        }

        public override void SetQuadraticTerm()
        {
            bool changed = false;
            if (Solver.A == null || Solver.A.GetLength(0) != Values.Length)
            {
                Solver.A = new double[Values.Length, Values.Length];
                changed = true;
            }

            if (KernelMatrix == null || changed)
                KernelMatrix = ComputeKernelMatrix();

            for (int i = 0; i < Values.Length; ++i)
                for (int j = 0; j < Values.Length; ++j)
                    Solver.A[i, j] = Labels[i] * Labels[j] * KernelMatrix[i, j];
        }
    }
}
