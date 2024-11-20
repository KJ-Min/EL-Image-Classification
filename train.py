import argparse  # 커맨드 라인 인자 파싱용

import numpy as np
from tqdm import tqdm  # 진행바 표시용
from utils._utils import make_data_loader  # 데이터 로더 생성용
from model import BaseModel  # 모델 정의용, 기본 모델 (현재는 사용하지 않음)

import torch
import torch.nn as nn  # edited
from torchvision.models import resnet18, ResNet18_Weights  # 사전학습된 ResNet18 모델

import sys


# 정확도 계산 함수
def acc(pred, label):
    pred = pred.argmax(dim=-1)  # 가장 높은 확률을 가진 클래스 선택
    return torch.sum(pred == label).item()  # 정확히 예측한 개수 반환


# 학습 함수
def train(args, data_loader, model):
    """
    TODO: Change the training code as you need. (e.g. different optimizer, different loss function, etc.)
            You can add validation code. -> This will increase the accuracy.
    """
    # BCELoss: Binary Cross Entropy Loss
    # run.py에서는 Sigmoid()를 사용하지만, 여기서는 사용하지 않아 에러 발생
    # criterion = torch.nn.BCELoss()
    criterion = nn.CrossEntropyLoss()

    # SGD 옵티마이저 사용
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    # 에포크 반복
    for epoch in range(args.epochs):
        train_losses = []  # 각 배치의 손실값 저장
        train_acc = 0.0  # 정확도 누적값
        total = 0  # 전체 데이터 개수
        print(f"[Epoch {epoch+1} / {args.epochs}]")

        model.train()  # 학습 모드로 설정
        pbar = tqdm(
            data_loader,
            desc="Training",
            leave=True,
            position=0,
            file=sys.stdout,
            dynamic_ncols=True,
        )
        sys.stdout.reconfigure(line_buffering=True)

        # 배치 단위 학습
        for i, (x, y) in enumerate(pbar):
            image = x.to(args.device)  # 이미지를 GPU로
            label = y.to(args.device)  # 레이블을 GPU로
            optimizer.zero_grad()  # 그래디언트 초기화

            output = model(image)  # 모델 예측
            label = label.squeeze()  # 차원 축소
            loss = criterion(output, label)  # 손실 계산
            loss.backward()  # 역전파
            optimizer.step()  # 가중치 업데이트

            # 통계 기록
            train_losses.append(loss.item())
            total += label.size(0)
            train_acc += acc(output, label)

            # 진행상황 업데이트 (선택사항)
            pbar.set_postfix(
                {
                    "loss": f"{np.mean(train_losses):.3f}",
                    "acc": f"{100.*train_acc/total:.2f}%",
                }
            )

        # 에포크 단위 통계 계산
        epoch_train_loss = np.mean(train_losses)
        epoch_train_acc = train_acc / total

        # 결과 출력
        print(f"Epoch {epoch+1}")
        print(f"train_loss : {epoch_train_loss}")
        print("train_accuracy : {:.3f}".format(epoch_train_acc * 100))

        torch.save(model.state_dict(), f"{args.save_path}/model.pth")


if __name__ == "__main__":
    # 커맨드 라인 인자 설정
    parser = argparse.ArgumentParser(
        description="2024 DL Term Project"
    )  # 커맨드 라인 인자 설정
    parser.add_argument(
        "--save-path", default="checkpoints/", help="Model's state_dict"
    )  # 모델 저장 경로
    parser.add_argument(
        "--data", default="term_project_train_data/", type=str, help="data folder"
    )  # default 경로를 변경 # 데이터 경로
    args = parser.parse_args()

    # GPU 사용 가능 여부 확인 # GPU 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    num_classes = 2  # edited # 이진 분류 문제

    """
    TODO: You can change the hyperparameters as you wish.
            (e.g. change epochs etc.)
    """

    # hyperparameters # 하이퍼파라미터 설정
    args.epochs = 5
    args.learning_rate = 0.1
    args.batch_size = 64

    # check settings # 설정 출력
    print("==============================")
    print("Save path:", args.save_path)
    print("Using Device:", device)
    print("Number of usable GPUs:", torch.cuda.device_count())

    # Print Hyperparameter # 하이퍼파라미터 출력
    print("Batch_size:", args.batch_size)
    print("learning_rate:", args.learning_rate)
    print("Epochs:", args.epochs)
    print("==============================")

    # Make Data loader and Model # 데이터 로더 생성
    train_loader, _ = make_data_loader(args)

    # custom model
    # model = BaseModel()

    # torchvision model # ResNet18 모델 생성 및 설정
    model = resnet18(weights=ResNet18_Weights)

    # you have to change num_classes to 2
    num_features = (
        model.fc.in_features
    )  # edited # ResNet18의 마지막 fully connected layer의 입력 특성 수 (512)
    model.fc = nn.Linear(
        num_features, num_classes
    )  # edited # 출력층을 이진 분류용으로 변경
    model.to(device)
    print(model)
    # Training The Model # 모델 학습 시작
    train(args, train_loader, model)
