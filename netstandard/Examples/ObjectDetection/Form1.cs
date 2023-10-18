using ObjectONNX;

namespace ObjectDetection
{
    public partial class Form1 : Form
    {
        private readonly ObjectDetector _objectDetector;

        public Form1()
        {
            InitializeComponent();
            DragDrop += Form1_DragDrop;
            DragEnter += Form1_DragEnter;
            AllowDrop = true;

            _objectDetector = new ObjectDetector(0.3f, 0.4f, 0.5f, NonMaxSuppressionMode.Basic);
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
            // inference session
            var results = _objectDetector.Forward(image);
            using var g = Graphics.FromImage(image);
            using var font = new Font("Arial", 22);
            using var brush = new SolidBrush(Color.Yellow);
            using var pen = new Pen(brush, 2);

            for (int i = 0; i < results.Length; i++)
            {
                g.DrawRectangle(pen, results[i].Rectangle);
                g.DrawString($"{ObjectDetector.Labels[results[i].Id]}, {Math.Round(100 * results[i].Score, 2)}%", font, brush, results[i].Rectangle.Left, results[i].Rectangle.Top);
            }

            BackgroundImage = image;
        }
    }
}