import io.skymind.auth.AuthClient;
import io.skymind.modelproviders.history.client.ModelHistoryClient;
import io.skymind.skil.predict.client.PredictServiceClient;
import io.skymind.skil.service.model.ClassificationResult;
import io.skymind.skil.service.model.MultiClassClassificationResult;
import io.skymind.skil.service.model.Prediction;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.UUID;

public class TestMnist {
    public static void main(String... args) throws Exception {
        AuthClient authClient = new AuthClient("http://127.0.0.1:9008");
        authClient.login("admin", "admin");
        String authToken = authClient.getAuthToken();

        //ModelHistoryClient mhClient = new ModelHistoryClient("http://127.0.0.1:9100");

        String basePath = "http://localhost:9008/endpoints/demo/model/imsenet/default";
        //String basePath = "http://127.0.0.1:9008/endpoints/demo/model/mnist/default";
        //String basePath = "http://localhost:9008/endpoints/demo/model/sample-mnist-model/default";
        //String basePath = "http://localhost:9008/endpoints/demo/model/tfmodel/default";
        //String basePath = "http://localhost:9008/endpoints/demo/model/keras-mnist-model/default";

        //String basePath = "http://localhost:9601";
        PredictServiceClient client = new PredictServiceClient(basePath);
        client.setAuthToken(authToken);

        INDArray eye = Nd4j.zeros(1, 299, 299, 3);
        //INDArray eye = Nd4j.eye(28).reshape(1, 28 * 28);
        Prediction input = new Prediction(eye, "eye");

        for (int i = 0; i < 100; i++) {
            long start = System.currentTimeMillis();
            Prediction result = client.predictArray(input);
            long end = System.currentTimeMillis();

            System.out.println(result);
            System.out.println("Took " + (end - start) + " ms");
        }
        /*
        MnistDataSetIterator data = new MnistDataSetIterator(1, 100);

        while (data.hasNext()) {
            DataSet ds = data.next();
            INDArray req = ds.getFeatures();
            INDArray label = ds.getLabels();

            ClassificationResult classify = client.classify(new Prediction(req, UUID.randomUUID().toString()));
            int truth = Nd4j.argMax(label, -1).max(-1).data().getInt(0);
            int pred = classify.getResults()[0];

            System.out.println(
                    "True: " + truth +
                    " Predicted: " + pred +
                    " Match: " + ((truth == pred) ? "true" : "false")
            );
        }*/
    }
}
