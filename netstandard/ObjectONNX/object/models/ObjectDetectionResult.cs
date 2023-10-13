using System.Drawing;

namespace ObjectONNX
{
    public class ObjectDetectionResult
    {
        public float Score { get; set; }

        public string Label { get; set; }

        public Rectangle Rectangle { get; set; }
    }
}
