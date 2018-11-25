import com.google.common.base.Stopwatch;
import io.skymind.skil.daemon.client.SKILDaemonClient;
import io.skymind.skil.predict.client.PredictServiceClient;
import io.skymind.skil.service.model.ClassificationResult;
import io.skymind.skil.service.model.Prediction;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.util.UUID;
import java.util.concurrent.TimeUnit;

public class TestMnist {
    public static void main(String... args) throws Exception {
        String host = "localhost:9008";
        String user = "admin";
        String pass = "admin";
        String modelName = "sample-mnist-model";

        if (args.length > 0) {
            host = args[0];
        }
        if (args.length > 1) {
            user = args[1];
        }
        if (args.length > 2) {
            pass = args[2];
        }
        if (args.length > 3) {
            modelName = args[3];
        }

        SKILDaemonClient authClient = new SKILDaemonClient("http://" + host);
        authClient.login(user, pass);
        String authToken = authClient.getAuthToken();


        String basePath = "http://" + host + "/endpoints/demo/model/" + modelName + "/default";

        PredictServiceClient client = new PredictServiceClient(basePath);
        client.setAuthToken(authToken);

        //INDArray black = Nd4j.zeros(1, 299, 299, 3);
        INDArray eye = Nd4j.eye(28).reshape(1, 28 * 28);
        Prediction input = new Prediction(eye, "eye");

        ClassificationResult result = client.classify(input);

        System.out.println(result);

        MnistDataSetIterator data = new MnistDataSetIterator(1, 1000);

        int total = 0;
        int right = 0;

        Stopwatch stopwatch = Stopwatch.createStarted();
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

            total++;
            if (truth == pred) {
                right++;
            }

        }

        long elapsed = stopwatch.elapsed(TimeUnit.SECONDS);
        float acc = (float)right / total;

        System.out.printf("Accuraccy %.4f", acc);
        System.out.println();
        System.out.println("Took " + elapsed + " seconds. (" + (float)total / elapsed + " rps)");
    }
}
