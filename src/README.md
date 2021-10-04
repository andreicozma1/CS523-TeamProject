# Source Code

## Pathway #2: AI as an Application: 
- Building machine learning models for a particular domain setting and evaluating them appropriately, e.g. predicting /r/AITA votes with data from Reddit. Projects in this pathway should:
1. Find or collect a [dataset](../datasets) for machine learning use.
2. Implement and train models with the dataset, i.e. with [scikit-learn](https://scikit-learn.org/) or [Keras](https://keras.io/).
3. Design an [evaluation](evaluations) that compares the performance of multiple models.

## Libraries
1. **Keras**
   - API Reference: https://keras.io/api/
   - Developer Guides: https://keras.io/guides/
   - Examples: https://keras.io/examples/
2. **Scikit-learn**
   - Getting Started: https://scikit-learn.org/stable/getting_started.html
   - Pre-Processing: https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing
   - Model Selection & Evaluation: https://scikit-learn.org/stable/model_selection.html#model-selection
   - Supervised Learning: https://scikit-learn.org/stable/supervised_learning.html
   - Unsupervised Learning: https://scikit-learn.org/stable/unsupervised_learning.html

## Environment Details
- Nvidia CUDA Toolkit: https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local
- NVIDIA cuDNN: https://developer.nvidia.com/cudnn
  - Download: https://developer.nvidia.com/rdp/cudnn-download
  - The NVIDIA CUDA® Deep Neural Network library (cuDNN) is a GPU-accelerated library of primitives for deep neural networks. cuDNN provides highly tuned implementations for standard routines such as forward and backward convolution, pooling, normalization, and activation layers. 
  - Deep learning researchers and framework developers worldwide rely on cuDNN for high-performance GPU acceleration. It allows them to focus on training neural networks and developing software applications rather than spending time on low-level GPU performance tuning. cuDNN accelerates widely used deep learning frameworks, including Caffe2, Chainer, Keras, MATLAB, MxNet, PaddlePaddle, PyTorch, and TensorFlow. For access to NVIDIA optimized deep learning framework containers that have cuDNN integrated into frameworks, visit NVIDIA GPU CLOUD to learn more and get started.
  - Developer Guide: https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html