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

            _objectDetector = new ObjectDetector(0.25f, 0.5f, ObjectDetectionModel.SSDInceptionV2);
            var image = new Bitmap("example.png", false);
            Process(image);
            BackgroundImage = image;
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
            BackgroundImage = image;
            Cursor = Cursors.Default;
        }

        private void Process(Bitmap image)
        {
            // params
            using var font = new Font("Arial", 22);

            // inference session
            var results = _objectDetector.Forward(image);

            using (var g = Graphics.FromImage(image))
            {
                for (int i = 0; i < results.Length; i++)
                {
                    // python rectangle
                    var result = results[i];
                    var c = Color.Yellow;
                    using var brush = new SolidBrush(c);
                    using var pen = new Pen(c) { Width = 3 };
                    g.DrawString(ObjectDetector.Labels[result.Id], font, brush, result.Rectangle.Left, result.Rectangle.Top);
                    g.DrawRectangle(pen, result.Rectangle);
                }
            }

            BackgroundImage = image;
        }
    }
}