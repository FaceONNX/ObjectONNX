using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using UMapx.Core;
using UMapx.Imaging;

namespace ObjectONNX
{
    /// <summary>
    /// Using for NonMaxSuppression operations.
    /// </summary>
    internal static class NonMaxSuppressionExensions
    {
        /// <summary>
        /// Get current protos from results
        /// </summary>
        /// <param name="sessionProtos">Array without repeats of all possible protos</param>
        /// <param name="results">Collection of results (predictions)</param>
        /// <param name="yoloSquare">Yolo square</param>
        /// <param name="classes">Count of classes</param>
        /// <returns>Results</returns>
        private static List<string> GetProtos(this string[] sessionProtos, List<float[]> results, int yoloSquare, int classes)
        {
            var predictions = results.ToArray();
            var lenght = predictions.Length;
            var protos = new List<string>();

            for (int i = 0; i < lenght; i++)
            {
                var prediction = predictions[i];
                var labels = new float[classes];
                for (int j = 0; j < classes; j++)
                {
                    labels[j] = prediction[j + yoloSquare];
                }
                _ = Matrice.Max(labels, out int argmax);
                protos.Add(sessionProtos[argmax]);
            }
            return protos;
        }

        /// <summary>
        /// Agnostic NMS filtration (without regard classes of recognized objects)
        /// </summary>
        /// <param name="results">Results</param>
        /// <param name="nmsThreshold">Threshold</param>
        /// <returns>Results</returns>
        public static List<float[]> AgnosticNMSFiltration(this List<float[]> results, float nmsThreshold)
        {
            var list = results.OrderByDescending(x => x[4]).ToList();
            var length = list.Count;

            for (int i = 0; i < length; i++)
            {
                var first = list[i];

                for (int j = i + 1; j < length; j++)
                {
                    var second = list[j];

                    var iou = Rectangles.IoU(
                        Rectangle.FromLTRB(
                        (int)first[0],
                        (int)first[1],
                        (int)first[2],
                        (int)first[3]),

                        Rectangle.FromLTRB(
                        (int)second[0],
                        (int)second[1],
                        (int)second[2],
                        (int)second[3]
                        ));

                    if (iou > nmsThreshold)
                    {
                        list.RemoveAt(j);
                        length = list.Count;
                        j--;
                    }
                }
            }
            return list;
        }

        /// <summary>
        /// NMS filtration (with regard classes of recognized objects). Classical algorithm.
        /// Filtration use in limits of one class
        /// </summary>
        /// <param name="results">Results</param>
        /// <param name="nmsThreshold">Threshold</param>
        /// <param name="sessionProtos">Array without repeats of all possible protos</param>
        /// <param name="yoloSquare">Yolo square</param>
        /// <param name="classes">Count of classes</param>
        /// <returns>Results</returns>
        public static List<float[]> NMSFiltration(
            this List<float[]> results,
            float nmsThreshold,
            string[] sessionProtos,
            int yoloSquare,
            int classes)
        {
            var list = results.OrderByDescending(x => x[4]).ToList();
            var length = list.Count;

            var tmpProtos = new List<string>();
            tmpProtos = GetProtos(sessionProtos, list, yoloSquare, classes);

            var classesLists = new Dictionary<string, List<float[]>>();

            for (int i = 0; i < length; i++)
            {
                var item = list[i];
                var proto = tmpProtos[i];

                if (classesLists.ContainsKey(proto))
                {
                    classesLists[proto].Add(item);
                }
                else
                {
                    classesLists.Add(proto, new List<float[]>());
                    classesLists[proto].Add(item);
                }
            }

            foreach (var classList in classesLists)
            {
                classList.Value.OrderByDescending(x => x[4]).ToList();
                length = classList.Value.Count;
                for (int i = 0; i < length; i++)
                {
                    var first = classList.Value[i];

                    for (int j = i + 1; j < length; j++)
                    {
                        var second = classList.Value[j];

                        var iou = Rectangles.IoU(
                            Rectangle.FromLTRB(
                            (int)first[0],
                            (int)first[1],
                            (int)first[2],
                            (int)first[3]),

                            Rectangle.FromLTRB(
                            (int)second[0],
                            (int)second[1],
                            (int)second[2],
                            (int)second[3]
                            ));

                        if (iou > nmsThreshold)
                        {
                            classList.Value.RemoveAt(j);
                            length = classList.Value.Count;
                            j--;
                        }
                    }
                }
            }

            list.Clear();
            foreach (var classList in classesLists)
            {
                foreach (var item in classList.Value)
                {
                    list.Add(item);
                }
            }
            return list;
        }

        /// <summary>
        /// Resize method with preserving proportions.
        /// </summary>
        /// <param name="image">Image</param>
        /// <param name="size">Size</param>
        /// <param name="value">Background value</param>
        /// <param name="interpolationMode">Interpolation mode</param>
        /// <returns>Image</returns>
        public static float[,] Resize(this float[,] image, Size size, float value, InterpolationMode interpolationMode = InterpolationMode.Bicubic)
        {
            int width = image.GetLength(1);
            int height = image.GetLength(0);
            int max = Math.Max(width, height);
            var rect = new Rectangle((max - width) / 2, (max - height) / 2, width, height);
            var temp = new float[max, max].Add(value);

            for (int y = 0; y < rect.Height; y++)
            {
                for (int x = 0; x < rect.Width; x++)
                {
                    temp[y + rect.Y, x + rect.X] = image[y, x];
                }
            }

            return temp.Resize(size.Height, size.Width, interpolationMode);
        }
    }
}
