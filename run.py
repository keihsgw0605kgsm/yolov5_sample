import torch
import hydra
import cv2
import os

@hydra.main(config_path="config", config_name="main")
def main(cfg):
    img_path = cfg.img_path
    repo = cfg.repo
    model_name = cfg.model_name

    model = torch.hub.load(repo, model_name)
    results = model(img_path)
    results.show()


if __name__ == "__main__":
    main()