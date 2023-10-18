using System;
using System.Drawing;

namespace ObjectONNX
{
    /// <summary>
    /// Defines object detector interface.
    /// </summary>
    public interface IObjectDetector : IDisposable
    {
        #region Interface

        /// <summary>
        /// Gets or sets detection threshold.
        /// </summary>
        public float DetectionThreshold { get; set; }

        /// <summary>
        /// Gets or sets confidence threshold.
        /// </summary>
        float ConfidenceThreshold { get; set; }

        /// <summary>
        /// Gets or sets NonMaxSuppression threshold.
        /// </summary>
        float NmsThreshold { get; set; }

        /// <summary>
        /// Gets or sets NonMaxSuppression mode.
        /// </summary>
        public NonMaxSuppressionMode NonMaxSuppressionMode { get; set; }

        /// <summary>
        /// Returns face detection results.
        /// </summary>
        /// <param name="image">Bitmap</param>
        /// <returns>Object detection result</returns>
        ObjectDetectionResult[] Forward(Bitmap image);

        /// <summary>
        /// Returns face detection results.
        /// </summary>
        /// <param name="image">Image in BGR terms</param>
        /// <returns>Object detection result</returns>
        ObjectDetectionResult[] Forward(float[][,] image);

        #endregion
    }
}
