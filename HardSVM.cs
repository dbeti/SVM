using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SVM
{
    /// <summary>
    /// Implementacija SVM-a s tvrdom marginom.
    /// </summary>
    public class HardSVM : SVM
    {
        public HardSVM(string name, double[][] values, double[] labels, IKernel kernel, string category) : base(name, values, labels, kernel, category)
        {
        }

        public double B { get; set; }

        public double Lambda { get; set; }

        public double Margin { get; set; }

        public override string AlgorithmName => "Hard";

        public override void Algorithm()
        {
            SupportVectors.Clear();
            SupportVectors.Capacity = 0;
            // Save support vectors indexes
            for (int i = 0; i < Solver.lagrangians.Length; ++i)
                if (Solver.lagrangians[i] > 0)
                    SupportVectors.Add(i);

            foreach (var i in SupportVectors)
            {
                foreach (var j in SupportVectors)
                {
                    Lambda += Labels[i] * Labels[j] * Solver.lagrangians[i] * Solver.lagrangians[j] * KernelMatrix[i, j];
                }
            }

            int chosen = SupportVectors.FirstOrDefault();
            B = Labels[chosen] * Lambda;

            foreach (var i in SupportVectors)
                B -= Solver.lagrangians[i] * Labels[i] * KernelMatrix[chosen, i];

            Margin = Math.Sqrt(Lambda);

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
                Solver.ubnd[i] = Double.PositiveInfinity;
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
