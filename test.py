import argparse  # 커맨드 라인 인자 파싱용
import torch  # PyTorch 딥러닝 프레임워크
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn  # edited
from tqdm import tqdm  # 진행률 표시바
from utils._utils import make_data_loader  # 커스텀 데이터 로더 생성 함수
from model import BaseModel  # 커스텀 모델 (주석처리됨)
from torchvision.models import resnet18  # 사전학습된 ResNet18 모델
import sys

sys.stdout.reconfigure(line_buffering=True)


# 테스트 함수 정의
def test(args, data_loader, model):
    """
    모델 성능을 평가하는 테스트 함수
    true: 실제 레이블을 저장할 배열
    pred: 예측값을 저장할 배열
    """
    true = np.array([])
    pred = np.array([])

    model.eval()  # 모델을 평가 모드로 설정

    # 테스트 데이터 반복 # 진행바 설정 수정
    pbar = tqdm(
        data_loader,
        desc="Testing",
        leave=True,
        position=0,
        file=sys.stdout,
        dynamic_ncols=True,
    )

    for i, (x, y) in enumerate(pbar):
        image = x.to(args.device)  # 이미지를 GPU/CPU로 이동
        label = y.to(args.device)  # 레이블을 GPU/CPU로 이동

        output = model(image)  # 모델 추론

        # 예측값 처리
        label = label.squeeze()
        output = output.argmax(dim=-1)  # 가장 높은 확률의 클래스 선택
        output = output.detach().cpu().numpy()
        pred = np.append(pred, output, axis=0)

        # 실제 레이블 처리
        label = label.detach().cpu().numpy()
        true = np.append(true, label, axis=0)

        # 진행바 업데이트를 위한 출력 버퍼 비우기
        sys.stdout.flush()

    return pred, true


# 메인 실행 부분
if __name__ == "__main__":
    # 커맨드 라인 인자 설정
    parser = argparse.ArgumentParser(description="2024 DL Term Project")
    parser.add_argument(
        "--model-path", default="checkpoints/model.pth", help="Model's state_dict"
    )
    parser.add_argument(
        "--data", default="term_project_train_data/", type=str, help="data folder"
    )
    args = (
        parser.parse_args()
    )  # 커맨드 라인에서 입력받은 인자들을 파싱하여 args 객체로 저장

    # GPU 사용 가능 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    num_classes = 2  # 이진 분류 문제

    # hyperparameters # 배치 사이즈 설정
    args.batch_size = 4

    # Make Data loader and Model  # 데이터 로더 생성 (utils/_utils.py의 함수 사용)
    _, test_loader = make_data_loader(args)

    # instantiate model
    # model = BaseModel()
    # 모델 생성
    # BaseModel 대신 ResNet18 사용
    model = resnet18()

    # ResNet18의 마지막 완전연결층을 2개 클래스로 수정
    num_features = (
        model.fc.in_features
    )  # ResNet18의 마지막 완전연결층의 입력 특성 수를 가져옴 (기본값: 512) # edited
    model.fc = nn.Linear(num_features, num_classes)  # edited

    # 저장된 모델 가중치 로드
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)  # 모델을 GPU/CPU로 이동

    # Test The Model
    # 모델 테스트 실행 - test() 함수를 호출하여 예측값(pred)과 실제값(true)을 반환받음
    pred, true = test(args, test_loader, model)

    # 정확도 계산 및 출력
    accuracy = (true == pred).sum() / len(
        pred
    )  # 예측값과 실제값을 비교하여 정확도 계산 (맞은 개수 / 전체 개수)
    print("Test Accuracy : {:.5f}".format(accuracy))
