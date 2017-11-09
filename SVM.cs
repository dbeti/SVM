using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SVM
{
    /// <summary>
    /// Apstraktna klasa koja sadrži metode i svojstva za implementaciju SVM algoritama. 
    /// </summary>
    public abstract class SVM
    {
        /// <summary>
        /// Ime problema. Prema njemu će se dohvaćati i generirati datoteke.
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// Specifikacija imena algoritma. Npr. ukoliko se isti problem rjesava pomocu SVM-a
        /// s tvrdom i mekom marginom, primjer SpecificName bi bio "Hard" i "Soft"
        /// </summary>
        public abstract string AlgorithmName { get; }

        /// <summary>
        /// Vektori učenja koji će se koristiti pri generiranju SVM modela.
        /// </summary>
        public double[][] Values { get; set; }

        /// <summary>
        /// Vrijednosti vektora učenja koji se koriste za generiranje SVM modela.
        /// </summary>
        public double[] Labels { get; set; }

        /// <summary>
        /// Jezgrina matrica čiji (i,j)-ti element sadrži evaluiranu vrijednost jezgrine funkcije
        /// za vektore Values[i] i Values[j].
        /// </summary>
        public double[,] KernelMatrix { get; set; }

        /// <summary>
        /// Nakon treniranja modela lista sadrži indekse vektora iz svojstva <see cref="Values"/> koji
        /// predstavljaju potporne vektore u SVM-u. (Oni koji imaju nenegativne odgovarajuće Lagrangeove 
        /// multiplikatore).
        /// </summary>
        public List<int> SupportVectors { get; set; } = new List<int>();

        /// <summary>
        /// Jezgrina funkcija koja će se koristiti u algoritmu.
        /// </summary>
        public IKernel Kernel { get; set; }

        /// <summary>
        /// Instanca QPSolver klase pomoću koje rješavamo optimizacijski problem u SVM-u.
        /// </summary>
        public QPSolver Solver { get; set; }

        /// <summary>
        /// U metodi je potrebno postaviti matricu A u <see cref="Solver"/>.
        /// </summary>
        public abstract void SetQuadraticTerm();

        /// <summary>
        /// U metodi je potrebno postaviti vektor b iz <see cref="Solver"/>.
        /// </summary>
        public abstract void SetLinearTerm();

        /// <summary>
        /// U metodi je potrebno postaviti ograničenja <see cref="QPSolver.lbnd"/>, <see cref="QPSolver.ubnd"/>,
        /// <see cref="QPSolver.c"/>, <see cref="QPSolver.ct"/> te vektor skaliranja <see cref="QPSolver.s"/>.
        /// </summary>
        public abstract void SetConstraints();

        /// <summary>
        /// U metodi je potrebno implementirati SVM algoritam, dakle sve ono što ide nakon dobivenog rješenja
        /// optimizacijskog problema.
        /// </summary>
        public abstract void Algorithm();

        /// <summary>
        /// Funkcija odlučivanja za SVM.  
        /// </summary>
        /// <param name="input">Vektor nad kojim želimo pokrenuti istrenirani SVM.</param>
        /// <returns>Ovisno o implementaciji SVM-a, npr. ukoliko se radi o klasifikaciji, -1 ukoliko 
        /// vektor ne pripada klasi za koju smo trenirali SVM, 1 inače.</returns>
        public abstract double Decide(double[] input);

        /// <summary>
        /// Treniranje SVM-a.
        /// </summary>
        public void Train()
        {

            SetQuadraticTerm();

            SetLinearTerm();

            SetConstraints();

            Solver.Initialize();

            Solver.Solve();

            Algorithm();

        }

        public SVM(String name, double[][] values, double[] labels, IKernel kernel, string category)
        {
            Values = values;
            Labels = labels;
            Kernel = kernel;
            Name = name;

            Solver = new QPSolver(values.Length, $"{Name}-{Kernel.Name}-{category}");
        }

        /// <summary>
        /// Metoda računa jezgrinu matricu za vektore učenja <see cref="Values"/>. Matrica se sprema u datoteku
        /// radi kasnije upotrebe ako ona ne postoji <see cref="KernelMatrix"/>, inače se pročita iz datoteke.
        /// </summary>
        /// <returns>Matrica čiji je (i,j)-ti element vrijednost jezgrine funkcije za vektore Values[i] and Values[j]</returns>
        protected double[,] ComputeKernelMatrix()
        {
            double[,] gram = new double[Values.Length, Values.Length];

            string path = $"{Directory.GetCurrentDirectory()}\\data\\{Name}-{Kernel.Name}-kernelmatrix.txt";

            // Ukoliko postoji jezgrina matrica za taj problem i algoritam pročitaj je s diska.
            if (File.Exists(path))
            {
                using (StreamReader file = new StreamReader(path))
                {
                    string line; int row = 0; string[] splitted;
                    while ((line = file.ReadLine()) != null)
                    {
                        splitted = line.Split(' ');

                        for (int j = 0; j < splitted.Length; ++j)
                            gram[row, j] = Convert.ToDouble(splitted[j]);

                        ++row;
                    }
                }
            }
            else //Inače je izračunaj i spremi u datoteku.
            {
                for (int i = 0; i < gram.GetLength(0); ++i)
                {
                    for (int j = 0; j < gram.GetLength(1); ++j)
                    {
                        if (gram[j, i] != 0)
                            gram[i, j] = gram[j, i];
                        else
                            gram[i, j] = Kernel.Compute(Values[i], Values[j]);
                    }
                }

                //Spremi matricu u datoteku. 
                using (StreamWriter file = new StreamWriter(path))
                {
                    StringBuilder line = new StringBuilder("");
                    for (int i = 0; i < gram.GetLength(0); ++i)
                    {
                        line.Clear();
                        for (int j = 0; j < gram.GetLength(1); ++j)
                        {
                            if (j == 0)
                                line.Append(gram[i, j].ToString());
                            else
                                line.Append(" " + gram[i, j].ToString());
                        }
                        file.WriteLine(line);
                    }
                }
            }
            return gram;
        }


        /// <summary>
        /// Klasa s kojom se rješava optimizacijski problem u SVM-u. Radi njezinog boljeg razumijevanja
        /// potrebno je pogledati dokumentaciju za ALGLIB biblioteku čije se metode interno koriste.
        /// </summary>
        public class QPSolver
        {
            // Vektori potrebni za postavljanje optimizacijskog problema pomoću metoda iz ALGLIB.
            public double[] lagrangians;

            public double[,] A;

            public double[] l;

            public double[] lbnd;

            public double[] ubnd;

            public double[,] c;

            public double[] s;

            public int[] ct;

            private alglib.minqpstate state;

            public alglib.minqpreport report;

            /// <summary>
            /// Ime problema koje se koristi kako bi se kreirala datoteka u koje se sprema/čita rješenje.
            /// </summary>
            public String Name { get; set; }

            public QPSolver(int size, string name)
            {
                Name = name;
                alglib.minqpcreate(size, out state);
            }

            public void Initialize()
            {
                alglib.minqpsetquadraticterm(state, A);
                alglib.minqpsetbc(state, lbnd, ubnd);
                alglib.minqpsetlc(state, c, ct);
                alglib.minqpsetscale(state, s);
            }

            /// <summary>
            /// Metoda rješava postavljeni optimizacijski problem pomoću metoda iz ALGLIB biblioteke.
            /// Za dano ime provjerava postoji li već datoteka. Ako postoji pročitat će rješenje iz nje,
            /// ukoliko ne postoji, rješist će problem te će spremiti rješenje u datoteku.
            /// </summary>
            public void Solve()
            {
                string path = $"{Directory.GetCurrentDirectory()}\\data\\{Name}.txt";

                if (File.Exists(path))
                {
                    using (StreamReader file = new StreamReader(path))
                    {
                        if (lagrangians == null || lagrangians.Length != A.GetLength(1))
                            lagrangians = new double[A.GetLength(1)];

                        string[] splitted = file.ReadLine().Split(' ');

                        for (int i = 0; i < splitted.Length; ++i)
                            lagrangians[i] = Convert.ToDouble(splitted[i]);

                    }
                }
                else
                {
                    alglib.minqpsetalgobleic(state, 0, 0, 0, 0);
                    alglib.minqpoptimize(state);
                    alglib.minqpresults(state, out lagrangians, out report);

                    if (report.terminationtype > 0)
                    {
                        using (StreamWriter file = new StreamWriter(path))
                        {
                            StringBuilder line = new StringBuilder("");
                            for (int i = 0; i < lagrangians.Length; ++i)
                            {
                                if (i == 0)
                                    line.Append(lagrangians[i].ToString());
                                else
                                    line.Append(" " + lagrangians[i].ToString());
                            }
                            file.WriteLine(line);
                        }
                    }
                }
            }

        }
    }
}
