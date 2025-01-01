import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt


def load_model(model_path):
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
    )

    # Load the state dict and remove the "module." prefix which occured during training(used "DataParallel" to use 2 GPUs)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[name] = v
    
    # Load the modified state dict
    model.load_state_dict(new_state_dict)
    model.eval()
    return model


def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    img = (img - 0.5) / 0.5 #to match the training scaling
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)
    return torch.tensor(img, dtype=torch.float32)


def segment_and_visualize(model, image_path):
    input_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_tensor) 
        output = torch.sigmoid(output).squeeze().numpy()

    binary_mask = (output > 0.5).astype(np.uint8)

    original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(original_img, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Predicted Mask")
    plt.imshow(binary_mask, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# Example usage
# model = load_model("tumor_segmentation_model.pth")
# img_path = "path/to/your/image.png"
# segment_and_visualize(model, img_path)
