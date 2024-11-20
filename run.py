import argparse  # 커맨드 라인 인자 처리
import os  # 파일/디렉토리 처리
import torch  # PyTorch 메인 라이브러리
from torch.utils.data import Dataset, DataLoader  # 데이터 처리 도구
from torchvision import datasets, transforms  # 이미지 처리 도구
from torchvision.models import resnet18, ResNet18_Weights  # 사전학습된 ResNet18 모델
from model import BaseModel  # 커스텀 모델 (현재는 사용하지 않음)
from tqdm import tqdm  # 진행바 표시
from PIL import Image  # 이미지 처리
import torch.nn as nn  # edited # 신경망 구성요소
import sys
from IPython.display import clear_output  # 추가

torch.manual_seed(0)  # 재현성을 위한 시드 설정


# 이진 분류를 위한 커스텀 데이터셋 클래스
class BinaryImageDataset(Dataset):
    def __init__(self, fault_dir, normal_dir, transform=None):
        """
        fault_dir: 불량 이미지 디렉토리 (라벨 0)
        normal_dir: 정상 이미지 디렉토리 (라벨 1)
        transform: 이미지 전처리 변환
        """
        self.fault_dir = fault_dir
        self.normal_dir = normal_dir
        self.transform = transform

        # 이미지 경로와 라벨을 리스트로 구성
        self.image_paths = [
            (os.path.join(fault_dir, f), 0) for f in os.listdir(fault_dir)
        ] + [(os.path.join(normal_dir, f), 1) for f in os.listdir(normal_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, label = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")  # RGB 이미지로 변환
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


# 정확도 계산 함수
def compute_accuracy(preds, labels):
    """
    preds: 모델의 예측값 (0~1 사이의 실수)
    labels: 실제 라벨 (0 또는 1)
    """
    preds = (preds >= 0.5).float()  # 0.5를 기준으로 이진 분류
    correct = (preds == labels).float().sum()
    return correct / len(preds)


# 모델 추론 함수
def inference(args, data_loader, model):
    model.eval()  # 평가 모드로 설정
    preds, labels = [], []

    with torch.no_grad():  # 그래디언트 계산 비활성화
        pbar = tqdm(data_loader, file=sys.stdout)  # file=sys.stdout 추가
        for images, label in pbar:
            images, label = images.to(args.device), label.to(args.device)
            output = model(images)
            preds.append(output.item())
            labels.extend(label.cpu().tolist())

            # 진행바 업데이트를 위한 출력 버퍼 비우기
            sys.stdout.flush()  # 추가

    # print(preds, labels)
    accuracy = compute_accuracy(torch.tensor(preds), torch.tensor(labels))
    return preds, accuracy


# 메인 실행 부분
if __name__ == "__main__":
    # 커맨드 라인 인자 설정
    parser = argparse.ArgumentParser(description="2024 DL Term Project")
    parser.add_argument(
        "--load-model", default="checkpoints/model.pth", help="Model's state_dict"
    )
    parser.add_argument("--batch-size", default=1, help="test loader batch size")
    parser.add_argument(
        "--fault-dir",
        default="term_project_train_data/fault",
        help="Directory for fault images",
    )
    parser.add_argument(
        "--normal-dir",
        default="term_project_train_data/normal",
        help="Directory for normal images",
    )

    args = parser.parse_args()

    # GPU 사용 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device

    # instantiate model
    # model = BaseModel()
    # model.load_state_dict(torch.load(args.load_model))
    # model.to(device)

    # torchvision models
    # model = resnet18(weights=None)

    # num_features = model.fc.in_features
    # model.fc = nn.Sequential(
    #     nn.Linear(num_features, 1),
    #     nn.Sigmoid()
    # )

    # model.load_state_dict(torch.load(args.load_model)) # ResNet18 모델 생성 및 설정
    model = resnet18(weights="IMAGENET1K_V1")
    num_features = model.fc.in_features
    # 이진 분류를 위한 출력층 수정
    model.fc = nn.Sequential(
        nn.Linear(num_features, 1),
        nn.Sigmoid(),  # 0~1 사이의 출력을 위한 시그모이드 활성화
    )

    model.to(device)  # GPU로 모델 이동
    # load dataset in test image folder
    # you may need to edit transform
    # 이미지 전처리 설정
    transform = transforms.Compose(
        [
            transforms.Resize((100, 200)),  # 이미지 크기 조정
            transforms.ToTensor(),  # 텐서 변환 및 정규화 (0~1)
        ]
    )

    # 테스트 데이터셋 및 데이터로더 생성
    test_data = BinaryImageDataset(args.fault_dir, args.normal_dir, transform=transform)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # write model inference # 모델 추론 실행
    preds, acc = inference(args, test_loader, model)

    # 결과 출력 및 저장
    print(f"Accuracy: {acc * 100:.2f}%")
    with open("result.txt", "w") as f:
        f.writelines("\n".join(map(str, preds)))
