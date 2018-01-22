import io.skymind.auth.AuthClient;
import io.skymind.skil.predict.client.PredictServiceClient;
import io.skymind.skil.service.model.ClassificationResult;
import io.skymind.skil.service.model.JsonArrayResponse;
import io.skymind.skil.service.model.MultiClassClassificationResult;
import io.skymind.skil.service.model.Prediction;
import org.apache.commons.io.IOUtils;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.image.data.Image;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ColorConversionTransform;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.InputStream;
import java.net.URL;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;

public class TestMnist {

    // https://github.com/thtrieu/darkflow/blob/master/cfg/yolo.cfg
    public static final int nClasses = 80;
    public static final int gridWidth = 19;
    public static final int gridHeight = 19;
    public static final double[][] priorBoxes = {{0.57273, 0.677385}, {1.87446, 2.06253}, {3.33843, 5.47434}, {7.88282, 3.52778}, {9.77052, 9.16828}};

    public static void main(String... args) throws Exception {
        List<String> labels = IOUtils.readLines(new URL("https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names").openStream());

        //AuthClient authClient = new AuthClient("http://localhost:9008");
        AuthClient authClient = new AuthClient("http://skildemo1.southcentralus.cloudapp.azure.com:9008");
        authClient.login("admin", "admin");
        String authToken = authClient.getAuthToken();

        //String basePath = "http://localhost:9008/endpoints/sfdemo/model/yolo/default";
        String basePath = "http://skildemo1.southcentralus.cloudapp.azure.com:9008/endpoints/yolo/model/yolo/default";
        //String basePath = "http://localhost:9008/endpoints/foo/model/bar/default";
        //String basePath = "http://localhost:9602";
        PredictServiceClient client = new PredictServiceClient(basePath);
        client.setAuthToken(authToken);

        InputStream dogStream = new URL("https://raw.githubusercontent.com/deeplearning4j/deeplearning4j/master/deeplearning4j-zoo/src/main/resources/goldenretriever.jpg").openStream();
        Mat dogMat = imdecode(new Mat(IOUtils.toByteArray(dogStream)), CV_LOAD_IMAGE_COLOR);
        NativeImageLoader imageLoader = new NativeImageLoader(608, 608, 3, new ColorConversionTransform(COLOR_BGR2RGB));
        INDArray dog = imageLoader.asMatrix(dogMat);
        dog = dog.permute(0, 2, 3, 1).muli(1.0 / 255.0).dup('c');
        Prediction input = new Prediction(dog, "dog");
        System.out.println(input.getPrediction().shapeInfoToString());

        //INDArray black = Nd4j.zeros(1, 608, 608, 3);
        //INDArray eye = Nd4j.eye(28).reshape(1, 28 * 28);
        //Prediction input = new Prediction(black, "black");

        INDArray boundingBoxPriors = Nd4j.create(priorBoxes);
        CanvasFrame frame = new CanvasFrame("TestMnist");
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

        long start = System.nanoTime();
        for (int i = 0; i < 1; i++) {
            JsonArrayResponse result = client.jsonArrayPredict(input);
            INDArray permuted = result.getArray().permute(0, 3, 1, 2);
            INDArray activated = YoloUtils.activate(boundingBoxPriors, permuted);
            List<DetectedObject> predictedObjects = YoloUtils.getPredictedObjects(boundingBoxPriors, activated, 0.6, 0.4);
            int width = dogMat.rows(), height = dogMat.cols();

            for (DetectedObject o : predictedObjects) {
                String label = labels.get(o.getPredictedClass());
                long x = Math.round(width  * o.getCenterX() / gridWidth);
                long y = Math.round(height * o.getCenterY() / gridHeight);
                long w = Math.round(width  * o.getWidth()   / gridWidth);
                long h = Math.round(height * o.getHeight()  / gridHeight);

                System.out.println("\"" + label + "\" at [" + x + "," + y + ";" + w + "," + h + "], conf = " + o.getConfidence());
                //System.out.println(o.toString());

                double[] xy1 = o.getTopLeftXY();
                double[] xy2 = o.getBottomRightXY();
                int x1 = (int) Math.round(width  * xy1[0] / gridWidth);
                int y1 = (int) Math.round(height * xy1[1] / gridHeight);
                int x2 = (int) Math.round(width  * xy2[0] / gridWidth);
                int y2 = (int) Math.round(height * xy2[1] / gridHeight);
                rectangle(dogMat, new Point(x1, y1), new Point(x2, y2), Scalar.RED);
                putText(dogMat, label, new Point(x1 + 2, y2 - 2), FONT_HERSHEY_DUPLEX, 1, Scalar.GREEN);
            }
            frame.setTitle("Dog - TestMnist");
            frame.setCanvasSize(width, height);
            frame.showImage(converter.convert(dogMat));
        }
        long end = System.nanoTime();

        System.out.println((end - start) / 1000000 + " ms");

        frame.waitKey();
    }

}
