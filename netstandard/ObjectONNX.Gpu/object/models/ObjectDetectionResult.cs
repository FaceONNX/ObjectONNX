﻿using System.Drawing;

namespace ObjectONNX
{
    /// <summary>
    /// Defines object detection result.
    /// </summary>
    public class ObjectDetectionResult
    {
        /// <summary>
        /// Gets or sets label id.
        /// </summary>
        public int Id { get; set; }

        /// <summary>
        /// Gets or sets score.
        /// </summary>
        public float Score { get; set; }

        /// <summary>
        /// Gets or sets rectangle.
        /// </summary>
        public Rectangle Rectangle { get; set; }

        /// <summary>
        /// Empty object detection result.
        /// </summary>
        public static ObjectDetectionResult Empty
        {
            get
            {
                return new ObjectDetectionResult
                {
                    Rectangle = Rectangle.Empty,
                    Score = 0,
                    Id = -1
                };
            }
        }
    }
}
