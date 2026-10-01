from pathlib import Path

import cv2
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import R3D_18_Weights
import torch.nn as nn

from torchvision.models.video import (
    r3d_18,
    R3D_18_Weights
)

# --------------------------------------------------
# JIGSAWS LOCATION
# --------------------------------------------------

JIGSAWS_ROOT = (
    Path.home()
    / "OneDrive"
    / "Documents"
    / "Robotic Surgery"
    / "JIGSAWS"
)


# --------------------------------------------------
# R3D-18 PREPROCESSING
# --------------------------------------------------

weights = R3D_18_Weights.DEFAULT
preprocess = weights.transforms()


# --------------------------------------------------
# GET 16 FRAMES FROM ONE GESTURE
# --------------------------------------------------

def get_video_clip(
    video_path,
    start_frame,
    end_frame,
    num_frames=16
):

    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    frame_numbers = np.linspace(
        start_frame,
        end_frame,
        num_frames,
        dtype=int
    )

    frames = []

    for frame_number in frame_numbers:

        video.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number - 1
        )

        success, frame = video.read()

        if not success:
            video.release()

            raise RuntimeError(
                f"Could not read frame {frame_number} "
                f"from {video_path}"
            )

        # BGR -> RGB
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frames.append(frame)

    video.release()

    # [T, H, W, C]
    frames = np.stack(frames)

    # NumPy -> PyTorch
    frames = torch.from_numpy(frames)

    # [T, H, W, C] -> [T, C, H, W]
    frames = frames.permute(0, 3, 1, 2)

    return frames


# --------------------------------------------------
# JIGSAWS PYTORCH DATASET
# --------------------------------------------------

class JIGSAWSDataset(Dataset):

    def __init__(self, root):

        self.root = Path(root)
        self.samples = []

        # Only use capture1 for now
        video_files = list(
            self.root.rglob("*_capture1.avi")
        )

        print(
            "Videos found:",
            len(video_files)
        )

        for video_path in video_files:

            # Example:
            # Suturing_B001_capture1.avi
            #
            # becomes:
            # Suturing_B001

            trial_name = video_path.name.replace(
                "_capture1.avi",
                ""
            )

            # Find matching transcription
            transcription_files = [
                path
                for path
                in self.root.rglob(
                    f"{trial_name}.txt"
                )
                if path.parent.name.lower()
                == "transcriptions"
            ]

            if len(transcription_files) == 0:

                print(
                    "⚠️ No transcription found for:",
                    trial_name
                )

                continue

            transcription_path = (
                transcription_files[0]
            )

            # Read gesture segments
            with open(
                transcription_path,
                "r"
            ) as file:

                for line in file:

                    parts = (
                        line.strip().split()
                    )

                    if len(parts) < 3:
                        continue

                    start_frame = int(
                        parts[0]
                    )

                    end_frame = int(
                        parts[1]
                    )

                    gesture = parts[2]

                    if not gesture.startswith("G"):
                        continue

                    # G1 -> 0
                    # G2 -> 1
                    # ...
                    # G15 -> 14

                    label = (
                        int(gesture[1:]) - 1
                    )

                    self.samples.append(
                        (
                            video_path,
                            start_frame,
                            end_frame,
                            label,
                            gesture
                        )
                    )

        print(
            "Gesture segments found:",
            len(self.samples)
        )


    def __len__(self):

        return len(self.samples)


    def __getitem__(self, index):

        (
            video_path,
            start_frame,
            end_frame,
            label,
            gesture
        ) = self.samples[index]

        clip = get_video_clip(
            video_path,
            start_frame,
            end_frame,
            num_frames=16
        )

        # Converts:
        #
        # [T, C, H, W]
        #
        # into R3D format:
        #
        # [C, T, 112, 112]

        clip = preprocess(clip)

        return clip, label
# --------------------------------------------------
# RUN TRAINING
# --------------------------------------------------

if __name__ == "__main__":

    # --------------------------------------------------
    # CREATE DATASET
    # --------------------------------------------------

    dataset = JIGSAWSDataset(JIGSAWS_ROOT)

    print("\n✅ DATASET CREATED")
    print("Total gesture examples:", len(dataset))

    clip, label = dataset[0]

    print("\n✅ FIRST SAMPLE")
    print("Clip shape:", clip.shape)
    print("Label:", label)
    print("Gesture:", f"G{label + 1}")


    # --------------------------------------------------
    # CREATE DATALOADER
    # --------------------------------------------------

    train_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )

    clips, labels = next(iter(train_loader))

    print("\n✅ BATCH CREATED")
    print("Batch shape:", clips.shape)
    print("Labels:", labels)


    # --------------------------------------------------
    # DEVICE
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("\nUsing device:", device)


    # --------------------------------------------------
    # LOAD R3D-18
    # --------------------------------------------------

    model = r3d_18(
        weights=R3D_18_Weights.DEFAULT
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        15
    )


    # --------------------------------------------------
    # FREEZE PRETRAINED LAYERS
    # --------------------------------------------------

    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.fc.parameters():
        parameter.requires_grad = True


    model = model.to(device)

    print("\n✅ MODEL READY")
    print(model.fc)


    # --------------------------------------------------
    # LOSS + OPTIMIZER
    # --------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.fc.parameters(),
        lr=0.001
    )


    # --------------------------------------------------
    # TRAIN FOR ONE EPOCH
    # --------------------------------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0


    for batch_number, (clips, labels) in enumerate(
        train_loader
    ):

        clips = clips.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(clips)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predictions = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()

        if (batch_number + 1) % 10 == 0:

            print(
                f"Batch {batch_number + 1} | "
                f"Loss: {loss.item():.4f}"
            )


    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    average_loss = (
        running_loss /
        len(train_loader)
    )

    training_accuracy = (
        100 * correct / total
    )


    print("\n✅ EPOCH COMPLETE")

    print(
        f"Average loss: "
        f"{average_loss:.4f}"
    )

    print(
        f"Training accuracy: "
        f"{training_accuracy:.2f}%"
    )


    # --------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------

    torch.save(
        model.state_dict(),
        "jigsaws_r3d18_test.pth"
    )

    print(
        "\n✅ MODEL SAVED:"
        " jigsaws_r3d18_test.pth"
    )