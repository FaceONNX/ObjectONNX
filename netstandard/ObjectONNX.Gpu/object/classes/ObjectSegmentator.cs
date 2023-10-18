using ObjectONNX.Properties;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Threading.Tasks;
using UMapx.Core;
using UMapx.Imaging;

namespace ObjectONNX
{
    /// <summary>
    /// Defines object segmentator.
    /// </summary>
    public class ObjectSegmentator : IObjectSegmentator
    {
        #region Private data

        /// <summary>
        /// Inference session.
        /// </summary>
        private readonly InferenceSession _session;

        #endregion

        #region Constructor

        /// <summary>
        /// Initializes object segmentator.
        /// </summary>
        public ObjectSegmentator()
        {
            _session = new InferenceSession(Resources.deeplabv3);
        }

        /// <summary>
        /// Initializes object segmentator.
        /// </summary>
        /// <param name="options">Session options</param>
        public ObjectSegmentator(SessionOptions options)
        {
            _session = new InferenceSession(Resources.deeplabv3, options);
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets or sets labels.
        /// </summary>
        public static readonly string[] Labels = new string[]
        {
            "Background",
            "Aeroplane",
            "Bicycle",
            "Bird",
            "Boat",
            "Bottle",
            "Bus",
            "Car",
            "Cat",
            "Chair",
            "Cow",
            "DiningTable",
            "Dog",
            "Horse",
            "Motorbike",
            "Person",
            "PottedPlant",
            "Sheep",
            "Sofa",
            "Train",
            "TV"
        };

        #endregion

        #region Methods

        /// <inheritdoc/>
        public int[,] Forward(Bitmap image)
        {
            var rgb = image.ToRGB(false);
            return Forward(rgb);
        }

        /// <inheritdoc/>
        public int[,] Forward(float[][,] image)
        {
            if (image.Length != 3)
                throw new ArgumentException("Image must be in BGR terms");

            // params
            var width = image[0].GetLength(1);
            var height = image[0].GetLength(0);
            var size = new Size(768, 768);
            var resized = new float[3][,];

            for (int i = 0; i < image.Length; i++)
            {
                resized[i] = image[i].ResizePreserved(size.Height, size.Width, 0.0f, InterpolationMode.Bicubic);
            }

            var dimentions = new int[] { 1, 3, size.Height, size.Width };
            var inputMeta = _session.InputMetadata;
            var name = inputMeta.Keys.ToArray()[0];

            // preprocessing
            var tensor = new DenseTensor<float>(dimentions);
            var mean = new[] { 0.485f, 0.456f, 0.406f }.Flip();
            var stddev = new[] { 0.229f, 0.224f, 0.225f }.Flip();

            // do job
            for (int i = 0; i < resized.Length; i++)
            {
                for (int y = 0; y < size.Height; y++)
                {
                    for (int x = 0; x < size.Width; x++)
                    {
                        // bgr to rgb and apply transform
                        tensor[0, resized.Length - i - 1, y, x] = (resized[i][y, x] - mean[i]) / stddev[i];
                    }
                }
            }

            // session run
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(name, tensor) };
            using var sessionResults = _session.Run(inputs);
            var results = sessionResults?.ToArray();

            // post-proccessing
            var count = Labels.Length;
            var masks = new float[count][,];
            var output = results[0].AsTensor<float>();

            // do job
            Parallel.For(0, count, k =>
            {
                var mask = new float[size.Height, size.Width];

                for (int j = 0; j < size.Height; j++)
                {
                    for (int i = 0; i < size.Width; i++)
                    {
                        mask[j, i] = output[0, k, j, i];
                    }
                }

                masks[k] = mask.ResizePreserved(height, width, InterpolationMode.Bicubic);
            });

            // results
            var max = new float[height, width];
            var ind = new int[height, width];
            var locker = new object();

            // do job parallel
            Parallel.For(0, count, k =>
            {
                for (int j = 0; j < height; j++)
                {
                    for (int i = 0; i < width; i++)
                    {
                        if (max[j, i] < masks[k][j, i])
                        {
                            lock (locker)
                            {
                                max[j, i] = masks[k][j, i];
                                ind[j, i] = k;
                            }
                        }
                    }
                }
            });

            // do job parallel
            Parallel.For(0, height, j =>
            {
                for (int i = 0; i < width; i++)
                {
                    for (int k = 0; k < count; k++)
                    {
                        if (max[j, i] < masks[k][j, i])
                        {
                            max[j, i] = masks[k][j, i];
                            ind[j, i] = k;
                        }
                    }
                }
            });

            return ind;
        }

        #endregion

        #region IDisposable

        private bool _disposed;

        /// <inheritdoc/>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        private void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    _session?.Dispose();
                }

                _disposed = true;
            }
        }

        /// <summary>
        /// Destructor.
        /// </summary>
        ~ObjectSegmentator()
        {
            Dispose(false);
        }

        #endregion
    }
}
