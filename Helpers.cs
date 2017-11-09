using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SVM
{
    public static class Helpers
    {
        /// <summary>
        /// Postavlja polje na zadanu vrijednost.
        /// </summary>
        /// <typeparam name="T">Tip polja.</typeparam>
        /// <param name="array">Polje za koje želimo zadati vrijednost.</param>
        /// <param name="value">Vrijednost koju zadajemo.</param>
        /// <returns>Vraća isto polje čiji su elementi postavljeni na zadanu vrijednost.</returns>
        public static T[] Populate<T>(this T[] array, T value)
        {
            for (int i = 0; i < array.Length; ++i)
                array[i] = value;

            return array;
        }

        /// <summary>
        /// Postavlja vrijednosti u parametar labels iz korpusa dokumenata prema zadanoj klasi.
        /// </summary>
        /// <param name="corpus">Korpus dokumenata predstavljen kao rječnik čiji su ključevi klase u kojima se dokumenti nalaze, a vrijednosti
        /// su liste dokumenata koji se nalaze u tim klasama.</param>
        /// <param name="labels">Lista u koju ćemo spremiti vrijednosti.</param>
        /// <param name="class">Ukoliko je dokument zadane klase u labels ćemo spremiti 1, -1 inače.</param>
        public static void Vectorize(ref Dictionary<string, List<Dictionary<string, int>>> corpus, ref List<double> labels, string @class)
        {
            foreach (var entry in corpus)
            {
                foreach (var document in entry.Value)
                {
                    if (entry.Key == @class)
                        labels.Add(1);
                    else
                        labels.Add(-1);
                }
            }
        }

        /// <summary>
        /// Čita dokumente za učenje iz datoteke name.
        /// </summary>
        /// <param name="corpus">Korpus dokumenata predstavljen kao rječnik čiji su ključevi klase u kojima se dokumenti nalaze, a vrijednosti
        /// su liste dokumenata koji se nalaze u tim klasama.</param>
        /// <param name="dictionary">Rječnik danog korpusa dokumenata.</param>
        /// <param name="name">Ime datoteke u kojoj se nalaze dokumenti. Jedna linija u datoteci predstavlja jedan dokument, prva riječ u liniji
        /// predstavlja klasu u kojoj se dokument nalazi, zatim slijedi tabulator te riječi dokumenta odvojene razmacima. Riječi ne sadrže inter-
        /// punkciju.</param>

        public static void ReadTextFile(ref Dictionary<string, List<Dictionary<string, int>>> corpus, ref HashSet<string> dictionary, string name)
        {
            string line;

            // Read the file and display it line by line.
            using (StreamReader file = new StreamReader($"{Directory.GetCurrentDirectory()}\\data\\{name}.txt"))
            {
                while ((line = file.ReadLine()) != null)
                {
                    char[] delimiters = new char[] { '\t' };
                    string[] parts = line.Split(delimiters, StringSplitOptions.RemoveEmptyEntries);

                    if (!corpus.ContainsKey(parts[0]))
                        corpus[parts[0]] = new List<Dictionary<string, int>>();

                    var words = parts[1].Split(' ');
                    var tmp = new Dictionary<string, int>();

                    foreach (var word in words)
                    {
                        dictionary.Add(word);
                        if (!tmp.ContainsKey(word))
                            tmp[word] = 1;
                        else
                            tmp[word] += 1;
                    }

                    corpus[parts[0]].Add(tmp);
                }

                file.Close();
            }
        }

        /// <summary>
        /// Pretvara dokumente iz korpusa u vektore koje će moći koristiti SVM, vektori su normalizirani i izračunati pomoću tf-idf.
        /// </summary>
        /// <param name="corpus">Korpus dokumenata predstavljen kao rječnik čiji su ključevi klase u kojima se dokumenti nalaze, a vrijednosti
        /// su liste dokumenata koji se nalaze u tim klasama.</param>
        /// <param name="dictionary">Rječnik korpusa, ključevi su riječi, a vrijednost svakog ključa je njezin redni broj u rječniku.</param>
        /// <param name="idf">Inverse document frequency matrica.</param>
        /// <param name="values">Lista vektora u koju spremamo novu reprezentaciju dokumenata.</param>
        public static void Vectorize(ref Dictionary<string, List<Dictionary<string, int>>> corpus, ref Dictionary<string, int> dictionary, ref Dictionary<string, double> idf, ref List<double[]> values)
        {
            foreach (var entry in corpus)
            {
                foreach (var document in entry.Value)
                {
                    double[] tmp = new double[dictionary.Count];
                    double norm = 0;

                    foreach (var word in document)
                    {
                        if (dictionary.ContainsKey(word.Key))
                        {
                            double value = word.Value * idf[word.Key];
                            norm += value * value;
                            tmp[dictionary[word.Key]] = value;
                        }
                    }
                    norm = Math.Sqrt(norm);

                    for (int i = 0; i < tmp.Length; ++i)
                        tmp[i] = tmp[i] / norm;

                    values.Add(tmp);
                }
            }

        }

        /// <summary>
        /// Računa inverse dokument frequency za svaku riječ u korpusu.
        /// </summary>
        /// <param name="corpus">Korpus dokumenata predstavljen kao rječnik čiji su ključevi klase u kojima se dokumenti nalaze, a vrijednosti
        /// su liste dokumenata koji se nalaze u tim klasama.</param>
        /// <param name="dictionary">Rječnik korpusa.</param>
        /// <returns>Vraća idf kao rječnik u kojem su ključevi riječi korpusa, a vrijednosti su odgovarajuće idf.</returns>
        public static Dictionary<string, double> Idf(List<Dictionary<string, int>> corpus, ref Dictionary<string, int> dictionary)
        {
            var idf = new Dictionary<string, double>();

            foreach (var word in dictionary)
            {
                int count = 0;
                foreach (var document in corpus)
                {
                    if (document.ContainsKey(word.Key))
                        ++count;
                }

                idf[word.Key] = Math.Log10(corpus.Count / (1.0 + count));
            }

            return idf;

        }
    }
}
