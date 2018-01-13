import io.skymind.auth.AuthClient;
import io.skymind.skil.predict.client.PredictServiceClient;
import io.skymind.skil.service.model.ClassificationResult;
import io.skymind.skil.service.model.JsonArrayResponse;
import io.skymind.skil.service.model.MultiClassClassificationResult;
import io.skymind.skil.service.model.Prediction;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

public class TestMnist {
    public static void main(String... args) throws Exception {
        AuthClient authClient = new AuthClient("http://localhost:9008");
        authClient.login("admin", "admin");
        String authToken = authClient.getAuthToken();

        String basePath = "http://localhost:9008/endpoints/sfdemo/model/yolo/default";
        //String basePath = "http://localhost:9008/endpoints/foo/model/bar/default";
        //String basePath = "http://localhost:9602";
        PredictServiceClient client = new PredictServiceClient(basePath);
        client.setAuthToken(authToken);

        INDArray black = Nd4j.zeros(1, 608, 608, 3);
        //INDArray eye = Nd4j.eye(28).reshape(1, 28 * 28);
        Prediction input = new Prediction(black, "black");

        long start = System.nanoTime();
        for (int i = 0; i < 1; i++) {
            JsonArrayResponse result = client.jsonArrayPredict(input);
            List<DetectedObject> predictedObjects = getPredictedObjects(result.getArray(), 0.6);

            for (DetectedObject o : predictedObjects) {
                System.out.println(o.toString());
            }
        }
        long end = System.nanoTime();

        System.out.println((end - start) / 1000000 + " ms");
    }

    public static List<DetectedObject> getPredictedObjects(INDArray networkOutput, double threshold){
        if(networkOutput.rank() != 4){
            throw new IllegalStateException("Invalid network output activations array: should be rank 4. Got array "
                    + "with shape " + Arrays.toString(networkOutput.shape()));
        }
        if(threshold < 0.0 || threshold > 1.0){
            throw new IllegalStateException("Invalid threshold: must be in range [0,1]. Got: " + threshold);
        }

        //Activations format: [mb, 5b+c, h, w]
        int mb = networkOutput.size(0);
        int h = networkOutput.size(1);
        int w = networkOutput.size(2);
        int b = 17;
        int c = (networkOutput.size(3)/b)-5;  //input.size(1) == b * (5 + C) -> C = (input.size(1)/b) - 5

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

        return out;
    }
}
