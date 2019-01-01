using System;
using Microsoft.ML;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Transforms.Conversions;
namespace ML2
{
    public class IrisData
    {
        public float SepalLength;
        public float SepalWidth;
        public float PetalLength;
        public float PetalWidth;
        public string Label;
    }
    public class AddData
    {
        public float Value1;
        public float Value2;
        public string Label;
    }
    public class AddPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabels;
    }
    public class IrisPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabels;
    }
    class MyEventArgs : EventArgs
    {
        private int _Value1, _Value2;

        public MyEventArgs(int Value1, int Value2) { _Value1 = Value1; _Value2 = Value2; }

        public int Value1 { get { return this._Value1; } }
        public int Value2 { get { return this._Value2; } }


    }
    class Publisher
    {
        
        private int _Value1;
        private int _Value2;
        
        public event EventHandler<MyEventArgs> Predictor;

        public int Value1 {

            set { this._Value1 = value; }
            get { return this._Value1; }
        }
        public int Value2
        {
            set { this._Value2 = value; this.Predictor(this, new MyEventArgs(_Value1, _Value2)); }
            get { return this._Value2; }
        }

        

    }

    
    class Program
    {
        private static TransformerChain<KeyToValueTransform> model;
        private static MLContext mlContext;
       

        private static bool MeasureAccuracy(string label, int value)
        {
            bool retVal = false;

            if (label.Contains("zero") && (value == 0)) retVal = true;
            if (label.Contains("one") && (value == 1)) retVal = true;
            if (label.Contains("two") && (value == 2)) retVal = true;
            if (label.Contains("three") && (value == 3)) retVal = true;
            if (label.Contains("four") && (value == 4)) retVal = true;
            if (label.Contains("five") && (value == 5)) retVal = true;
            if (label.Contains("six") && (value == 6)) retVal = true;
            if (label.Contains("seven") && (value == 7)) retVal = true;
            if (label.Contains("eight") && (value == 8)) retVal = true;
            if (label.Contains("nine") && (value == 9)) retVal = true;
            if (label.Contains("ten") && (value == 10)) retVal = true;
            if (label.Contains("eleven") && (value == 11)) retVal = true;
            if (label.Contains("twelve") && (value == 12)) retVal = true;
            if (label.Contains("thirteen") && (value == 13)) retVal = true;
            if (label.Contains("fourteen") && (value == 14)) retVal = true;
            if (label.Contains("fifteen") && (value == 15)) retVal = true;
            if (label.Contains("sixteen") && (value == 16)) retVal = true;
            if (label.Contains("seventeen") && (value == 17)) retVal = true;
            if (label.Contains("eighteen") && (value == 18)) retVal = true;


            return retVal;

        }

        public static void cb_Function(object sender,MyEventArgs args)
        {
    
            var prediction = model.MakePredictionFunction<AddData, AddPrediction>(mlContext).Predict(
               new AddData()
               {
                   Value1 = args.Value1,
                   Value2 =args.Value2,

               });

            bool accuracy = false;
        
            accuracy = MeasureAccuracy(prediction.PredictedLabels, args.Value1 + args.Value2);
            Console.WriteLine($"Arg1 {args.Value1} Arg2 {args.Value2} Number is: {prediction.PredictedLabels} Accuracy is {accuracy}");
        }
        static void Main(string[] args)
        {
            Publisher pub = new Publisher();
            pub.Predictor += cb_Function;  //delegate function
            


            mlContext= new MLContext();


            string dataPath = "add.txt";
            var reader = mlContext.Data.TextReader(new TextLoader.Arguments()
            {
                Separator = ",",
                HasHeader = false,
                Column = new[]
                {
                    new TextLoader.Column("Value1", DataKind.R4, 0),
                    new TextLoader.Column("Value2", DataKind.R4, 1),
                   new TextLoader.Column("Label", DataKind.Text, 2)
                }
            });

            IDataView trainingDataView = reader.Read(new MultiFileSource(dataPath));
            // STEP 3: Transform your data and add a learner
            // Assign numeric values to text in the "Label" column, because only
            // numbers can be processed during model training.
            // Add a learning algorithm to the pipeline. e.g.(What type of iris is this?)
            // Convert the Label back into original text (after converting to number in step 3)
            
            var pipeline = mlContext.Transforms.Categorical.MapValueToKey("Label")
                .Append(mlContext.Transforms.Concatenate("Features", "Value1", "Value2"))
                .Append(mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(label: "Label", features: "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
                
                //var pipeline=mlContext.Transforms.Concatenate("Features","Value1","Value2")
               // .Append(mlContext.BinaryClassification.Trainers.FastTree(numLeaves: 50, numTrees: 50, minDatapointsInLeafs: 20));



            // STEP 4: Train your model based on the data set  
            model = pipeline.Fit(trainingDataView);
            int paramValue1 = 0;
            int paramValue2 = 0;
            /*
            do
            {
                Console.WriteLine("Input :");
                paramValue1 = Int32.Parse(Console.ReadLine());
                paramValue2 = Int32.Parse(Console.ReadLine());

                pub.Value1 = paramValue1;
                pub.Value2 = paramValue2;
            } while ((paramValue1 != -1) || (paramValue2 != -1));
            */

            for (int i = 0; i <= 9; i++)
            {
                for (int j = 0; j <= 9; j++)
                {
                    paramValue1 = i;
                    paramValue2 = j;
                    pub.Value1 = paramValue1;
                    pub.Value2 = paramValue2;
                }
            }

            // STEP 5: Use your model to make a prediction
            // You can change these numbers to test different predictions


            /*
            string dataPath = "data.txt";
            var reader = mlContext.Data.TextReader(new TextLoader.Arguments()
            {
                Separator = ",",
                HasHeader = true,
                Column = new[]
                {
                    new TextLoader.Column("SepalLength", DataKind.R4, 0),
                    new TextLoader.Column("SepalWidth", DataKind.R4, 1),
                    new TextLoader.Column("PetalLength", DataKind.R4, 2),
                    new TextLoader.Column("PetalWidth", DataKind.R4, 3),
                    new TextLoader.Column("Label", DataKind.Text, 4)
                }
            });

            IDataView trainingDataView = reader.Read(new MultiFileSource(dataPath));
            // STEP 3: Transform your data and add a learner
            // Assign numeric values to text in the "Label" column, because only
            // numbers can be processed during model training.
            // Add a learning algorithm to the pipeline. e.g.(What type of iris is this?)
            // Convert the Label back into original text (after converting to number in step 3)
            var pipeline = mlContext.Transforms.Categorical.MapValueToKey("Label")
                .Append(mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
                .Append(mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(label: "Label", features: "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // STEP 4: Train your model based on the data set  
            var model = pipeline.Fit(trainingDataView);

            // STEP 5: Use your model to make a prediction
            // You can change these numbers to test different predictions
            var prediction = model.MakePredictionFunction<IrisData, IrisPrediction>(mlContext).Predict(
                new IrisData()
                {
                    SepalLength = 3.3f,
                    SepalWidth = 1.6f,
                    PetalLength = 0.2f,
                    PetalWidth = 5.1f,
                });

            Console.WriteLine($"Predicted flower type is: {prediction.PredictedLabels}");
            */

            //Console.WriteLine("Hello World!");
            Console.Read();
        }
    }
}
