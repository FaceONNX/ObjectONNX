using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using ObjectONNX.Properties;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using UMapx.Core;
using UMapx.Imaging;

namespace ObjectONNX
{
    /// <summary>
    /// Defines object detector.
    /// </summary>
    public class ObjectDetector : IObjectDetector
    {
        #region Private data
        /// <summary>
        /// Inference session.
        /// </summary>
        private readonly InferenceSession _session;
        #endregion

        #region Constructor

        /// <summary>
        /// Initializes face detector.
        /// </summary>
        /// <param name="confidenceThreshold">Confidence threshold</param>
        /// <param name="nmsThreshold">NonMaxSuppression threshold</param>
        public ObjectDetector(float confidenceThreshold = 0.95f, float nmsThreshold = 0.5f)
        {
            _session = new InferenceSession(Resources.ssd_inception_v2_coco);
            ConfidenceThreshold = confidenceThreshold;
            NmsThreshold = nmsThreshold;
        }

        /// <summary>
        /// Initializes face detector.
        /// </summary>
        /// <param name="options">Session options</param>
        /// <param name="confidenceThreshold">Confidence threshold</param>
        /// <param name="nmsThreshold">NonMaxSuppression threshold</param>
        public ObjectDetector(SessionOptions options, float confidenceThreshold = 0.95f, float nmsThreshold = 0.5f)
        {
            _session = new InferenceSession(Resources.ssd_inception_v2_coco, options);
            ConfidenceThreshold = confidenceThreshold;
            NmsThreshold = nmsThreshold;
        }

        #endregion

        #region Properties

        /// <inheritdoc/>
        public float ConfidenceThreshold { get; set; }

        /// <inheritdoc/>
        public float NmsThreshold { get; set; }

        #endregion

        #region Methods

        /// <inheritdoc/>
        public Rectangle[] Forward(Bitmap image)
        {
            var rgb = image.ToRGB(false);
            return Forward(rgb);
        }

        /// <inheritdoc/>
        public Rectangle[] Forward(float[][,] image)
        {
            if (image.Length != 3)
                throw new ArgumentException("Image must be in BGR terms");

            var width = image[0].GetLength(1);
            var height = image[0].GetLength(0);
            var dimentions = new int[] { 1, height, width, 3 };
            var inputMeta = _session.InputMetadata;
            var name = inputMeta.Keys.ToArray()[0];

            // preprocessing
            var tensors = image.ToByteTensor(true);
            var inputData = tensors.Merge(true);

            // session run
            var t = new DenseTensor<byte>(inputData, dimentions);
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(name, t) };
            using var outputs = _session.Run(inputs);
            var results = outputs.ToArray();
            var detection_boxes = results[0].AsTensor<float>();
            var detection_classes = results[1].AsTensor<float>();
            var detection_scores = results[2].AsTensor<float>();
            var num_detections = results[3].AsTensor<float>()[0];

            // post-proccessing
            var boxes_picked = new List<Rectangle>();

            for (int i = 0; i < num_detections; i++)
            {
                var score = detection_scores[0, i];

                if (score > ConfidenceThreshold)
                {
                    //var label = labels[(int)detection_classes[0, i] - 1];

                    var x = (int)(detection_boxes[0, i, 0] * height);
                    var y = (int)(detection_boxes[0, i, 1] * width);
                    var w = (int)(detection_boxes[0, i, 2] * height);
                    var h = (int)(detection_boxes[0, i, 3] * width);

                    // python rectangle
                    var rectangle = Rectangle.FromLTRB(y, x, h, w);
                }
            }

            // non-max suppression
            var length = boxes_picked.Count;

            for (int i = 0; i < length; i++)
            {
                var first = boxes_picked[i];

                for (int j = i + 1; j < length; j++)
                {
                    var second = boxes_picked[j];
                    var iou = first.IoU(second);

                    if (iou > NmsThreshold)
                    {
                        boxes_picked.RemoveAt(j);
                        length = boxes_picked.Count;
                        j--;
                    }
                }
            }

            return boxes_picked.ToArray();
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

        ~ObjectDetector()
        {
            Dispose(false);
        }

        #endregion
    }
}
