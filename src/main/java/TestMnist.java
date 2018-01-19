import io.skymind.auth.AuthClient;
import io.skymind.skil.predict.client.PredictServiceClient;
import io.skymind.skil.service.model.ClassificationResult;
import io.skymind.skil.service.model.JsonArrayResponse;
import io.skymind.skil.service.model.MultiClassClassificationResult;
import io.skymind.skil.service.model.Prediction;
import org.apache.commons.io.IOUtils;
import org.datavec.image.data.Image;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ColorConversionTransform;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Broadcast;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

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

        NativeImageLoader imageLoader = new NativeImageLoader(608, 608, 3, new ColorConversionTransform(COLOR_BGR2RGB));
        Image dogImage = imageLoader.asImageMatrix(new URL("https://raw.githubusercontent.com/deeplearning4j/deeplearning4j/master/deeplearning4j-zoo/src/main/resources/goldenretriever.jpg").openStream());
        int width = dogImage.getOrigW(), height = dogImage.getOrigH();
        INDArray dog = dogImage.getImage();
        dog = dog.permute(0, 2, 3, 1).muli(1.0 / 255.0).dup('c');
        Prediction input = new Prediction(dog, "dog");
        System.out.println(input.getPrediction().shapeInfoToString());

        //INDArray black = Nd4j.zeros(1, 608, 608, 3);
        //INDArray eye = Nd4j.eye(28).reshape(1, 28 * 28);
        //Prediction input = new Prediction(black, "black");

        long start = System.nanoTime();
        for (int i = 0; i < 1; i++) {
            JsonArrayResponse result = client.jsonArrayPredict(input);
            List<DetectedObject> predictedObjects = getPredictedObjects(result.getArray(), 0.6);

            for (DetectedObject o : predictedObjects) {
                String label = labels.get(o.getPredictedClass());
                long x = Math.round(width  * o.getCenterX() / gridWidth);
                long y = Math.round(height * o.getCenterY() / gridHeight);
                long w = Math.round(width  * o.getWidth()   / gridWidth);
                long h = Math.round(height * o.getHeight()  / gridHeight);

                System.out.println("\"" + label + "\" at [" + x + "," + y + ";" + w + "," + h + "], conf = " + o.getConfidence());
                //System.out.println(o.toString());
            }
        }
        long end = System.nanoTime();

        System.out.println((end - start) / 1000000 + " ms");
    }

    public static INDArray activate(INDArray input) {
        //Essentially: just apply activation functions...

        int mb = input.size(0);
        int h = input.size(2);
        int w = input.size(3);
        int b = priorBoxes.length;
        int c = (input.size(1)/b)-5;  //input.size(1) == b * (5 + C) -> C = (input.size(1)/b) - 5

        INDArray output = Nd4j.create(input.shape(), 'c');
        INDArray output5 = output.reshape('c', mb, b, 5+c, h, w);
        INDArray output4 = output;  //output.get(all(), interval(0,5*b), all(), all());
        INDArray input4 = input.dup('c');    //input.get(all(), interval(0,5*b), all(), all()).dup('c');
        INDArray input5 = input4.reshape('c', mb, b, 5+c, h, w);

        //X/Y center in grid: sigmoid
        INDArray predictedXYCenterGrid = input5.get(all(), all(), interval(0,2), all(), all());
        Transforms.sigmoid(predictedXYCenterGrid, false);

        //width/height: prior * exp(input)
        INDArray predictedWHPreExp = input5.get(all(), all(), interval(2,4), all(), all());
        INDArray predictedWH = Transforms.exp(predictedWHPreExp, false);
        Broadcast.mul(predictedWH, Nd4j.create(priorBoxes), predictedWH, 1, 2);  //Box priors: [b, 2]; predictedWH: [mb, b, 2, h, w]

        //Confidence - sigmoid
        INDArray predictedConf = input5.get(all(), all(), point(4), all(), all());   //Shape: [mb, B, H, W]
        Transforms.sigmoid(predictedConf, false);

        output4.assign(input4);

        //Softmax
        //TODO OPTIMIZE?
        INDArray inputClassesPreSoftmax = input5.get(all(), all(), interval(5, 5+c), all(), all());   //Shape: [minibatch, C, H, W]
        INDArray classPredictionsPreSoftmax2d = inputClassesPreSoftmax.permute(0,1,3,4,2) //[minibatch, b, c, h, w] To [mb, b, h, w, c]
                .dup('c').reshape('c', new int[]{mb*b*h*w, c});
        Transforms.softmax(classPredictionsPreSoftmax2d, false);
        INDArray postSoftmax5d = classPredictionsPreSoftmax2d.reshape('c', mb, b, h, w, c ).permute(0, 1, 4, 2, 3);

        INDArray outputClasses = output5.get(all(), all(), interval(5, 5+c), all(), all());   //Shape: [minibatch, C, H, W]
        outputClasses.assign(postSoftmax5d);

        return output;
    }

    public static double overlap(double x1, double x2, double x3, double x4) {
        if (x3 < x1) {
            if (x4 < x1) {
                return 0;
            } else {
                return Math.min(x2, x4) - x1;
            }
        } else {
            if (x2 < x3) {
                return 0;
            } else {
                return Math.min(x2, x4) - x3;
            }
        }
    }

    /** intersection over union */
    public static double iou(DetectedObject o1, DetectedObject o2) {
        double x1min  = o1.getCenterX() - o1.getWidth() / 2;
        double x1max  = o1.getCenterX() + o1.getWidth() / 2;
        double y1min  = o1.getCenterY() - o1.getHeight() / 2;
        double y1max  = o1.getCenterY() + o1.getHeight() / 2;

        double x2min  = o2.getCenterX() - o2.getWidth() / 2;
        double x2max  = o2.getCenterX() + o2.getWidth() / 2;
        double y2min  = o2.getCenterY() - o2.getHeight() / 2;
        double y2max  = o2.getCenterY() + o2.getHeight() / 2;

        double ow = overlap(x1min, x1max, x2min, x2max);
        double oh = overlap(y1min, y1max, y2min, y2max);

        double intersection = ow * oh;
        double union = o1.getWidth() * o1.getHeight() + o2.getWidth() * o2.getHeight() - intersection;
        return intersection / union;
    }

    /** non-maximum suppression */
    public static void nms(List<DetectedObject> objects) {
        for (DetectedObject o1 : objects) {
            for (DetectedObject o2 : objects) {
                if (o1.getPredictedClass() == o2.getPredictedClass()
                        && o1.getConfidence() < o2.getConfidence()
                        && iou(o1, o2) > 0.4) {
                    o1.setPredictedClass(nClasses);
                }
            }
        }
        Iterator<DetectedObject> it = objects.iterator();
        while (it.hasNext()) {
            if (it.next().getPredictedClass() == nClasses) {
                it.remove();
            }
        }
    }

    public static List<DetectedObject> getPredictedObjects(INDArray networkOutput, double threshold){
        if(networkOutput.rank() != 4){
            throw new IllegalStateException("Invalid network output activations array: should be rank 4. Got array "
                    + "with shape " + Arrays.toString(networkOutput.shape()));
        }
        if(threshold < 0.0 || threshold > 1.0){
            throw new IllegalStateException("Invalid threshold: must be in range [0,1]. Got: " + threshold);
        }
        System.out.println(networkOutput.shapeInfoToString());

        networkOutput = networkOutput.permute(0, 3, 1, 2);

        networkOutput = activate(networkOutput);

        //Activations format: [mb, 5b+c, h, w]
        int mb = networkOutput.size(0);
        int h = networkOutput.size(2);
        int w = networkOutput.size(3);
        int b = priorBoxes.length;
        int c = (networkOutput.size(1)/b)-5;  //input.size(1) == b * (5 + C) -> C = (input.size(1)/b) - 5

        //Reshape from [minibatch, B*(5+C), H, W] to [minibatch, B, 5+C, H, W] to [minibatch, B, 5, H, W]
        INDArray output5 = networkOutput.dup('c').reshape(mb, b, 5+c, h, w);
        INDArray predictedConfidence = output5.get(all(), all(), point(4), all(), all());    //Shape: [mb, B, H, W]
        INDArray softmax = output5.get(all(), all(), interval(5, 5+c), all(), all());

        List<DetectedObject> out = new ArrayList<>();
        for( int i=0; i<mb; i++ ){
            for( int x=0; x<w; x++ ){
                for( int y=0; y<h; y++ ){
                    for( int box=0; box<b; box++ ){
                        double conf = predictedConfidence.getDouble(i, box, y, x);
                        if(conf < threshold){
                            continue;
                        }

                        double px = output5.getDouble(i, box, 0, y, x); //Originally: in 0 to 1 in grid cell
                        double py = output5.getDouble(i, box, 1, y, x); //Originally: in 0 to 1 in grid cell
                        double pw = output5.getDouble(i, box, 2, y, x); //In grid units (for example, 0 to 13)
                        double ph = output5.getDouble(i, box, 3, y, x); //In grid units (for example, 0 to 13)

                        //Convert the "position in grid cell" to "position in image (in grid cell units)"
                        px += x;
                        py += y;


                        INDArray sm;
                        try (MemoryWorkspace wsO = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                            sm = softmax.get(point(i), point(box), all(), point(y), point(x)).dup();
                        }

                        out.add(new DetectedObject(i, px, py, pw, ph, sm, conf));
                    }
                }
            }
        }

        nms(out);
        return out;
    }
}
