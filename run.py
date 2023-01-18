import torch
import hydra


@hydra.main(config_path="config", config_name="main")
def main(cfg):
    img_path = cfg.img_path
    repo_name = cfg.repo_name
    model_name = cfg.model_name

    model = torch.hub.load(repo_name, model_name)

    results = model(img_path)

    results.show()


if __name__ == "__main__":
    main()