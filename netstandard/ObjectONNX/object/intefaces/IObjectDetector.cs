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
        /// Gets or sets confidence threshold.
        /// </summary>
        float ConfidenceThreshold { get; set; }

        /// <summary>
        /// Gets or sets NonMaxSuppression threshold.
        /// </summary>
        float NmsThreshold { get; set; }

        /// <summary>
        /// Returns object detection results.
        /// </summary>
        /// <param name="image">Bitmap</param>
        /// <returns>Rectangles</returns>
        Rectangle[] Forward(Bitmap image);

        /// <summary>
        /// Returns object detection results.
        /// </summary>
        /// <param name="image">Image in BGR terms</param>
        /// <returns>Rectangles</returns>
        Rectangle[] Forward(float[][,] image);

        #endregion
    }
}
