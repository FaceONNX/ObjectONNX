using ObjectONNX;
using UMapx.Imaging;

namespace PersonDetectionAndSegmentation
{
    public partial class Form1 : Form
    {
        private readonly ObjectSegmentator _objectSegmentator;
        private readonly ObjectDetector _objectDetector;

        public Form1()
        {
            InitializeComponent();

            DragDrop += Form1_DragDrop;
            DragEnter += Form1_DragEnter;
            AllowDrop = true;
            Text = "ObjectONNX: Person detection and segmentation";
            BackgroundImageLayout = ImageLayout.Zoom;

            _objectDetector = new ObjectDetector(0.3f, 0.4f, 0.5f, NonMaxSuppressionMode.Basic);
            _objectSegmentator = new ObjectSegmentator();

            var image = new Bitmap("example.png", false);
            Process(image);
        }

        private void Form1_DragEnter(object sender, DragEventArgs e)
        {
            e.Effect = e.Data.GetDataPresent(DataFormats.FileDrop) ? DragDropEffects.All : DragDropEffects.None;
        }

        private void Form1_DragDrop(object sender, DragEventArgs e)
        {
            Cursor = Cursors.WaitCursor;
            var file = ((string[])e.Data.GetData(DataFormats.FileDrop, true))[0];
            var image = new Bitmap(file, false);
            Process(image);
            Cursor = Cursors.Default;
        }

        private void Process(Bitmap image)
        {
            // get results
            var objectDetectionResults = _objectDetector.Forward(image).Where(x => x.Id == 0).ToArray();
            var segmentationResults = _objectSegmentator.Forward(image);
            
            // draw
            using var g = Graphics.FromImage(image);
            using var font = new Font("Arial", 22);
            using var brush = new SolidBrush(Color.FromArgb(255, 0, 255, 0));
            using var pen = new Pen(brush, 2);

            for (int i = 0; i < objectDetectionResults.Length; i++)
            {
                g.DrawRectangle(pen, objectDetectionResults[i].Rectangle);
                g.DrawString($"{ObjectDetector.Labels[objectDetectionResults[i].Id]}, {Math.Round(100 * objectDetectionResults[i].Score, 2)}%", 
                    font, brush, objectDetectionResults[i].Rectangle.Left, objectDetectionResults[i].Rectangle.Top);
            }

            var results = new float[image.Height, image.Width];

            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    results[y, x] = segmentationResults[y, x] == 15 ? 1 : 0;
                }
            }

            using var mask = results.FromGrayscale();
            var maskColorFilter = new MaskColorFilter(Color.FromArgb(128, Color.Yellow));
            maskColorFilter.Apply(image, mask);

            image.Save("output.png", System.Drawing.Imaging.ImageFormat.Png);

            BackgroundImage = image;
        }
    }
}