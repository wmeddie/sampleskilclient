import io.skymind.auth.AuthClient;
import io.skymind.modelproviders.history.client.ModelHistoryClient;
import io.skymind.skil.predict.client.PredictServiceClient;
import io.skymind.skil.service.model.ClassificationResult;
import io.skymind.skil.service.model.MultiClassClassificationResult;
import io.skymind.skil.service.model.Prediction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

public class TestMnist {
    public static void main(String... args) throws Exception {
        AuthClient authClient = new AuthClient("http://127.0.0.1:9008");
        authClient.login("admin", "admin");
        String authToken = authClient.getAuthToken();

        //ModelHistoryClient mhClient = new ModelHistoryClient("http://127.0.0.1:9100");

        //String basePath = "http://127.0.0.1:9008/endpoints/demo/model/mnist/default";
        String basePath = "http://localhost:9008/endpoints/demo/model/sample-mnist-model/default";
        //String basePath = "http://localhost:9601";
        PredictServiceClient client = new PredictServiceClient(basePath);
        client.setAuthToken(authToken);

        //INDArray black = Nd4j.zeros(1, 299, 299, 3);
        INDArray eye = Nd4j.eye(28).reshape(1, 28 * 28);
        Prediction input = new Prediction(eye, "eye");

        ClassificationResult result = client.classify(input);

        System.out.println(result);


    }
}
