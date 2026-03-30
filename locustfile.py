from locust import HttpUser, task, between
import io
from PIL import Image

class MLPredictionUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        """
        Creates a dummy image in memory to use for the load test requests,
        avoiding file I/O bottleneck during simulation.
        """
        img = Image.new('RGB', (150, 150), color = (73, 109, 137))
        self.img_bytes = io.BytesIO()
        img.save(self.img_bytes, format='JPEG')

    @task
    def predict_endpoint(self):
        self.img_bytes.seek(0)
        files = {
            'file': ('test_image.jpg', self.img_bytes.read(), 'image/jpeg')
        }
        self.client.post("/predict", files=files)

    @task(3)
    def health_endpoint(self):
        self.client.get("/health")
