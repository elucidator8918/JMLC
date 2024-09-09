# FER Evaluation: Explainable Evaluation Framework for Facial Expression Recognition in Web-Based Learning Environments

This repository contains the code and scripts associated with the paper titled **"Explainable Evaluation Framework For Facial Expression Recognition in Web-Based Learning Environments"**. The repository is organized into two main folders: `Adv` (Advanced Fine-Tuning) and `Naive` (Basic Fine-Tuning).

## Repository Structure

```
.
├── Adv
│   ├── InceptionV3.py
│   ├── MobileNetV2.py
│   ├── ResNet50.py
│   ├── run.sh
│   └── VIT.py
├── Naive
│   ├── inception_v3.py
│   ├── mobilenetv2.py
│   └── resnet_50.py
└── README.md
```

### 1. **Adv (Advanced Fine-Tuning)**

The `Adv` folder contains scripts for advanced fine-tuning of various models. These scripts include techniques such as transfer learning with more sophisticated hyperparameter tuning, data augmentation, and other state-of-the-art training strategies aimed at improving the performance and explainability of Facial Expression Recognition (FER) models. They were on run on A100 GPU.

- **InceptionV3.py**: Advanced training script for the InceptionV3 model.
- **MobileNetV2.py**: Advanced training script for the MobileNetV2 model.
- **ResNet50.py**: Advanced training script for the ResNet50 model.
- **VIT.py**: Advanced training script for the Vision Transformer (VIT) model.

For the advanced fine tuning scripts except `VIT.py`, run the following shell file. 

```bash
bash run.sh
```

For `VIT.py`, look into the first 2 comments of the script!

### 2. **Naive (Basic Fine-Tuning)**

The `Naive` folder contains scripts for basic fine-tuning of the models. These scripts serve as a baseline, implementing straightforward transfer learning without extensive tuning or optimization. They are useful for comparison against the advanced fine-tuning techniques.

- **inception_v3.py**: Basic fine-tuning script for the InceptionV3 model.
- **mobilenetv2.py**: Basic fine-tuning script for the MobileNetV2 model.
- **resnet_50.py**: Basic fine-tuning script for the ResNet50 model.

## Paper Summary

This repository supports the research presented in the paper **"Explainable Evaluation Framework For Facial Expression Recognition in Web-Based Learning Environments"**. The study introduces an explainable evaluation framework for FER models, with a focus on improving model transparency and performance in the context of online learning environments. 

### Key Contributions:
- **Explainability**: Techniques to enhance the interpretability of FER models, making them more understandable to educators and learners.
- **Advanced Fine-Tuning**: The `Adv` scripts incorporate cutting-edge methods to optimize model performance, including detailed training strategies and adjustments.
- **Comparison Framework**: The `Naive` scripts provide a baseline to evaluate the effectiveness of advanced fine-tuning approaches.

## Results and Evaluation

Detailed results and evaluations of the models trained using these scripts can be found in the accompanying research paper. The scripts are designed to be easily modifiable, allowing researchers to experiment with different architectures and fine-tuning strategies.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For any questions or inquiries, please reach out to the author of the paper, Prof. Amira Mouakher.