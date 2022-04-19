import cv2
import numpy as np
import grpc
import lprnet_pb2, lprnet_pb2_grpc

image_size = (94, 24)

if __name__ == "__main__":
    image = cv2.imread("/home/pji/img_box.jpg")
    img = cv2.resize(image, image_size, interpolation=cv2.INTER_CUBIC)
    img = img.astype("float32")
    img -= 127.5
    img *= 0.0078125
    img = np.transpose(img, (2, 0, 1))
    # print(img)
    # print(np.frombuffer(np.ndarray.tobytes(img), dtype=np.float32).reshape(3,24,94))
    conn=grpc.insecure_channel('localhost:50052')
    client = lprnet_pb2_grpc.LprnetServiceStub(channel=conn)
    request = lprnet_pb2.LprnetRequest(images=[np.ndarray.tobytes(img)])
    response = client.predict(request)
    for label in response.labels:
        print(label)
