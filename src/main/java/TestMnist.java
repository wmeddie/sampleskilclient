import io.skymind.auth.AuthClient;
import io.skymind.skil.predict.client.PredictServiceClient;
import io.skymind.skil.service.model.ClassificationResult;
import io.skymind.skil.service.model.JsonArrayResponse;
import io.skymind.skil.service.model.MultiClassClassificationResult;
import io.skymind.skil.service.model.Prediction;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;

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
            //System.out.println(result.toString());

            //DetectedObject o = new DetectedObject()
        }
        long end = System.nanoTime();

        System.out.println((end - start) / 1000000 + " ms");
    }
}
