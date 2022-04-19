FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN pip install grpcio protobuf -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY app /workspace/

ENTRYPOINT python lprnet_service.py
