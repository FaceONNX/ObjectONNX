using System.Drawing;

namespace ObjectONNX
{
    /// <summary>
    /// Defines object detection result.
    /// </summary>
    public class ObjectDetectionResult
    {
        /// <summary>
        /// Gets or sets score.
        /// </summary>
        public float Score { get; set; }

        /// <summary>
        /// Gets or sets label.
        /// </summary>
        public string Label { get; set; }

        /// <summary>
        /// Gets or sets rectangle.
        /// </summary>
        public Rectangle Rectangle { get; set; }
    }
}
