/*import com.mashape.unirest.http.Unirest;
import io.skymind.skil.daemon.client.SKILDaemonClient;
import io.skymind.skil.predict.client.PredictServiceClient;
import io.skymind.skil.service.model.ClassificationResult;
import io.skymind.skil.service.model.Prediction;
import org.datavec.spark.transform.client.DataVecTransformClient;
import org.datavec.spark.transform.model.Base64NDArrayBody;
import org.datavec.spark.transform.model.SingleCSVRecord;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.serde.base64.Nd4jBase64;

import java.util.Arrays;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

public class TestSalesforce {
    public static void main(String... args) throws Exception {
        SKILDaemonClient authClient = new SKILDaemonClient("http://127.0.0.1:9008");
        authClient.login("admin", "admin123");
        String authToken = authClient.getAuthToken();

        //ModelHistoryClient mhClient = new ModelHistoryClient("http://127.0.0.1:9100");

        DataVecTransformClient transformClient = new DataVecTransformClient("http://localhost:9008/endpoints/demo/datavec/sftp/default");
        Unirest.setDefaultHeader("Authorization", "Bearer " + authToken);

        //                            Ammount, Probability, ExpectedRev, Type,           Lead Source      Fiscal Quarter
        String[] arg = new String[] { "15000", "10",        "1500",      "New Customer", "Purchased List", "1" };
        SingleCSVRecord singleCSVRecord = transformClient.transformIncremental(new SingleCSVRecord(arg));
        Base64NDArrayBody base64NDArrayBody = transformClient.transformArrayIncremental(new SingleCSVRecord(arg));
        List<Float> ndarray = singleCSVRecord.getValues().stream().map(Float::parseFloat).collect(Collectors.toList());
        INDArray array = Nd4j.create(ndarray);
        INDArray array2 = Nd4jBase64.fromBase64(base64NDArrayBody.getNdarray());

        String basePath = "http://127.0.0.1:9008/endpoints/demo/model/sfmodel/default";
        //String basePath = "http://localhost:9008/endpoints/foo/model/sample-mnist-model/default";
        //String basePath = "http://localhost:9601";
        PredictServiceClient client = new PredictServiceClient(basePath);
        client.setAuthToken(authToken);

        //INDArray black = Nd4j.zeros(1, 299, 299, 3);
        Prediction input = new Prediction(array, UUID.randomUUID().toString());

        ClassificationResult result = client.classify(input);

        System.out.println(result);
    }
}
**/

public class TestSalesforce {
}