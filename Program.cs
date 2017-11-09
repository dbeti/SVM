using SVM;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SVM
{
    class Program
    {
        /// <summary>
        /// Program pokreće implementirane SVM-ove prema zadanim parametrima.
        /// </summary>
        /// <param name="args">Prvi parametar predstavlja ime problema prema kojem se dohvaćaju
        /// odgovarajuće datoteke za učenje i testiranje. Npr. za "r8-all" dohvaća datoteke
        /// "r8-all-train.txt" i "r8-all-test.txt". Datoteke se moraju nalazi u direktoriju "data"
        /// koji se nalazi u istoj hijerahiji kao i .exe datoteka programa.
        /// 
        /// Drugi parametar određuje tip SVM implementacije. Mogući parametri su "hard" i "soft" 
        /// za SVM s tvrdom i mekom marginom.
        /// 
        /// Treći parametar određuje koja će se jezgra koristiti u SVM algoritmima. Mogući parametri su
        /// "polynomial" i "gauss".
        /// 
        /// Četvrti parametar je zapravo polje argumenata za jezgre. Tako za polinomnu jezgru dajemo
        /// dva parametra, stupanj i odmak polinoma, a za Gaussovu dajemo vrijednost sigma parametra.
        /// 
        /// </param>
        static void Main(string[] args)
        {
            if (args.Length < 4)
            {
                Console.WriteLine("Usage: problem_name SVM_type kernel kernel_arguments");
            }
            else
            {
                List<double[]> testValues = new List<double[]>();
                List<double> testLabels = new List<double>();

                var trainingCorpus = new Dictionary<string, List<Dictionary<string, int>>>();
                var testCorpus = new Dictionary<string, List<Dictionary<string, int>>>();

                var trainingSet = new HashSet<string>();
                var testSet = new HashSet<string>();

                String name = args[0];
                String type = args[1];
                String kernel = args[2];
                List<double> arguments = new List<double>();

                for (var i = 3; i < args.Length; ++i)
                    arguments.Add(Convert.ToDouble(args[i]));

                Helpers.ReadTextFile(ref trainingCorpus, ref trainingSet, $"{name}-train");
                Helpers.ReadTextFile(ref testCorpus, ref testSet, $"{name}-test");

                var dictionary = trainingSet.OrderBy(x => x).Select((x, i) => new { Key = x, Value = i }).ToDictionary(x => x.Key, x => x.Value);

                var trainingIdf = Helpers.Idf(trainingCorpus.SelectMany(c => c.Value).ToList(), ref dictionary);

                List<double[]> trainingValues = new List<double[]>();
                List<double> trainingLabels = new List<double>();

                Helpers.Vectorize(ref trainingCorpus, ref dictionary, ref trainingIdf, ref trainingValues);
                Helpers.Vectorize(ref testCorpus, ref dictionary, ref trainingIdf, ref testValues);

                SVM svm;
                IKernel iKernel;

                if (kernel == "polynomial")
                    iKernel = new PolynomialKernel(arguments.ToArray());
                else if (kernel == "gauss")
                    iKernel = new GaussKernel(arguments.FirstOrDefault());
                else
                    throw new ArgumentException("Cannot recognize kernel type!");

                if (type == "hard")
                    svm = new HardSVM(name, trainingValues.ToArray(), trainingLabels.ToArray(), iKernel, "");
                else if (type == "soft")
                    svm = new SoftSVM(name, trainingValues.ToArray(), trainingLabels.ToArray(), iKernel, "");
                else
                    throw new ArgumentException("Cannot recognize SVM type!");

                string path = $"{Directory.GetCurrentDirectory()}\\data\\{svm.Name}-{svm.AlgorithmName}-result-{svm.Name}-{svm.Kernel.Name}.txt";

                double tp, fp, fn, tn;

                foreach (var @class in trainingCorpus)
                {
                    double nu = 1.0;
                    svm.Solver.Name = $"{svm.Name}-{svm.AlgorithmName}-{svm.Kernel.Name}-{@class.Key}";

                    using (StreamWriter file = new StreamWriter(path, true))
                    {
                        StringBuilder sb = new StringBuilder();
                        // Line containes class,tp,tn,fp,fn,precision,recall,accuracy,F1
                        sb.Append(@class.Key);

                        Console.WriteLine($"=============={@class.Key}==============");
                        Console.WriteLine($"=============={System.DateTime.Now}==============");



                        trainingLabels.Clear();
                        testLabels.Clear();
                        Helpers.Vectorize(ref trainingCorpus, ref trainingLabels, @class.Key);
                        Helpers.Vectorize(ref testCorpus, ref testLabels, @class.Key);

                        int positive = 0, negative = 0;
                        for (int i = 0; i < trainingLabels.Count; ++i)
                        {
                            positive += (trainingLabels[i] == 1) ? 1 : 0;
                            negative += (trainingLabels[i] == -1) ? 1 : 0;
                        }

                        Console.WriteLine($"{@class.Key} training samples: {positive} of {positive + negative}");

                        positive = 0; negative = 0;
                        for (int i = 0; i < testLabels.Count; ++i)
                        {
                            positive += (testLabels[i] == 1) ? 1 : 0;
                            negative += (testLabels[i] == -1) ? 1 : 0;
                        }

                        Console.WriteLine($"{@class.Key} test samples: {positive} of {positive + negative}");

                        svm.Values = trainingValues.ToArray();
                        svm.Labels = trainingLabels.ToArray();

                        if (svm is SoftSVM)
                        {
                            // Find best parameter nu
                            do
                            {
                                nu -= 0.005;
                                (svm as SoftSVM).Nu = nu;
                                svm.Train();
                            } while (svm.Solver.report != null && svm.Solver.report.terminationtype == -3);
                        }
                        else
                        {
                            svm.Train();
                        }

                        Console.WriteLine($"nu {nu}");
                        sb.Append($"\t{nu}");

                        tp = 0; tn = 0; fp = 0; fn = 0;

                        for (int i = 0; i < testValues.Count; ++i)
                        {
                            double decided = svm.Decide(testValues[i]);

                            if (testLabels[i] == 1)
                            {
                                if (decided == testLabels[i])
                                    tp += 1;
                                else
                                    fn += 1;
                            }
                            else
                            {
                                if (decided == testLabels[i])
                                    tn += 1;
                                else
                                    fp += 1;
                            }
                        }
                        double precision = tp / (tp + fp);
                        double recall = tp / (tp + fn);

                        double accuracy = (tp + tn) / (tp + tn + fn + fp);
                        double F1 = 2 * (precision * recall) / (precision + recall);

                        Console.WriteLine($"tp tn fp fn : {tp} {tn} {fp} {fn}");
                        sb.Append($"\t{tp}\t{tn}\t{fp}\t{fn}");

                        Console.WriteLine($"Precision/recall: {precision} {recall}");
                        sb.Append($"\t{precision}\t{recall}");

                        Console.WriteLine("Accuracy: {0} || F1: {1}", accuracy * 100, F1);
                        sb.Append($"\t{accuracy * 100}\t{F1 * 100}");

                        Console.WriteLine($"=======================================");
                        file.WriteLine(sb);
                        file.Close();
                    }


                }
                System.Console.ReadLine();
            }

        }
    }
}
