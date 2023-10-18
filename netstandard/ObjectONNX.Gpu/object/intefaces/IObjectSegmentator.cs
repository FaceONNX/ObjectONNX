using System;
using System.Drawing;

namespace ObjectONNX
{
    /// <summary>
    /// Defines object segmentator interface.
    /// </summary>
    public interface IObjectSegmentator : IDisposable
    {
        #region Interface

        /// <summary>
        /// Returns object segmentation results.
        /// </summary>
        /// <param name="image">Bitmap</param>
        /// <returns>Result</returns>
        int[,] Forward(Bitmap image);

        /// <summary>
        /// Returns object segmentation results.
        /// </summary>
        /// <param name="image">Image in BGR terms</param>
        /// <returns>Result</returns>
        int[,] Forward(float[][,] image);

        #endregion
    }
}
