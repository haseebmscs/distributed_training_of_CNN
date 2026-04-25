import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from config import DATA_ROOT, BATCH_SIZE, DATA_LOADER_WORKERS

# Training transform — with augmentation
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
])

# Test transform — no augmentation
TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
])

class CIFAR10Dataset(Dataset):
    """
    Custom Dataset for CIFAR-10.
    Reads images from organised folders:
        dataset/data/train/airplane/
        dataset/data/train/cat/
        ...
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir  = root_dir
        self.transform = transform
        self.samples   = []

        # Get sorted class names
        self.classes = sorted(os.listdir(root_dir))

        # Build (image_path, label) pairs
        for class_name in self.classes:
            class_folder = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_folder):
                continue
            label = self.classes.index(class_name)
            for file_name in os.listdir(class_folder):
                if file_name.lower().endswith(('.png','.jpg','.jpeg')):
                    img_path = os.path.join(class_folder, file_name)
                    self.samples.append((img_path, label))

        print(f"[Dataset] Loaded {len(self.samples)} images "
              f"from {len(self.classes)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

def get_dataloaders():
    train_dir = os.path.join(DATA_ROOT, "train")
    test_dir  = os.path.join(DATA_ROOT, "test")

    train_dataset = CIFAR10Dataset(train_dir, TRAIN_TRANSFORM)
    test_dataset  = CIFAR10Dataset(test_dir,  TEST_TRANSFORM)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=DATA_LOADER_WORKERS,
        pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=DATA_LOADER_WORKERS,
        pin_memory=False
    )

    print(f"[Dataset] Train batches: {len(train_loader)}")
    print(f"[Dataset] Test  batches: {len(test_loader)}")

    return train_loader, test_loader

    
def _download_and_organise():
    """
    Downloads CIFAR-10 and organises into class folders.
    Runs automatically if organised data is missing.
    """
    import shutil

    raw_dir   = os.path.join(DATA_ROOT, "_raw")
    train_dst = os.path.join(DATA_ROOT, "train")
    test_dst  = os.path.join(DATA_ROOT, "test")

    CLASSES = [
        "airplane","automobile","bird","cat","deer",
        "dog","frog","horse","ship","truck"
    ]

    # Download via torchvision
    raw_train = torchvision.datasets.CIFAR10(
        root=raw_dir, train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    raw_test = torchvision.datasets.CIFAR10(
        root=raw_dir, train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    to_pil = transforms.ToPILImage()

    def save_split(dataset, dst_dir):
        # Create class folders
        for cls in CLASSES:
            os.makedirs(os.path.join(dst_dir, cls), exist_ok=True)

        print(f"[Dataset] Saving {len(dataset)} images to {dst_dir}...")
        for idx, (img_tensor, label) in enumerate(dataset):
            cls_name  = CLASSES[label]
            file_path = os.path.join(dst_dir, cls_name, f"{idx:05d}.png")
            to_pil(img_tensor).save(file_path)
            if (idx+1) % 10000 == 0:
                print(f"  Saved {idx+1}/{len(dataset)}")

    save_split(raw_train, train_dst)
    save_split(raw_test,  test_dst)
    print("[Dataset] Organisation complete!")


if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders()

    images, labels = next(iter(train_loader))
    print(f"Batch shape : {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"First label : {labels[0].item()}")