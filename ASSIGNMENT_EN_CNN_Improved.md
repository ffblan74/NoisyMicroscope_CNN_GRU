# A noisy microscope — CNN classification and GRU image restoration

**A3 practical · PyTorch · 3 hours · individual or pairs**

## Scenario and learning goals

You are prototyping two tools for a microscopy teaching lab: identifying a cell category and restoring a noisy colour image. Learn to design a CNN, represent an image as a sequence, train a unidirectional GRU and assess results critically. This practical concerns educational image processing; its results have no clinical validity.

BloodMNIST provides RGB images of eight cell categories: basophil, eosinophil, erythroblast, immature granulocytes, lymphocyte, monocyte, neutrophil and platelet. See [MedMNIST](https://medmnist.com/v2) and DATA_LICENSE.txt for attribution and CC BY 4.0 licensing.

The supplied balanced subset contains 3,200 training, 640 validation and 800 test images, selected separately from the official partitions. Image shape is 28×28×3. Convert uint8 pixels to float values in [0,1], with PyTorch shape (B,3,28,28). Both models use the same partitions. Cell labels are used only for classification.

## 1. Explore the data — 15 minutes

Run the supplied setup and loading cells. Display sample images with cell names. Explain batch, channel and spatial dimensions. Describe why training, validation and test sets have distinct roles.

## 2. Classify cells using a CNN — 65 minutes

Spend the first 10 minutes researching BatchNorm2d and Dropout in the official PyTorch documentation. Explain what they do, why they may help, their train/eval behaviour, the meaning of dropout probability 0.25, and their trainable parameters. Distinguish learned BatchNorm parameters from running-statistic buffers. Explain why no_grad does not replace model.eval. Cite your sources. Their implementation is supplied in the notebook; your explanations are required.

Complete a model with Conv2d(3,32,3,padding=1), BatchNorm2d(32), ReLU, MaxPool2d(2), Conv2d(32,64,3,padding=1), BatchNorm2d(64), ReLU, MaxPool2d(2), flatten, Linear(64×7×7,128), ReLU, Dropout(0.25), and Linear(128,8) producing eight logits. Fixed input rescaling to [-1,1] is applied inside the CNN. Batch normalisation and dropout code are supplied. Include biases.

Before using PyTorch's automatic counter, calculate every output shape and the parameter count of each layer. Use Cout×(Cin×k²+1) for convolution and dout×(din+1) for a linear layer. Batch normalisation adds two trainable values per channel (scale and shift). Running means and variances are buffers. Explain why pooling, ReLU and dropout add no trainable parameters.

Complete the gradient-update helper, cross-entropy loss and Adam optimiser (learning rate 0.001, weight decay 0.0001). Train for 30 epochs. The supplied training loop keeps the best validation checkpoint. Plot losses and report test accuracy, comparing it with uniform random guessing.

Compute and display an **8×8 confusion matrix**, using readable cell names. True classes are rows; predicted classes are columns. Check the total count, recover accuracy using the diagonal and identify the two largest off-diagonal confusions. Explain what a row-normalised matrix would reveal. Display some predicted images and analyse an error.

## 3. Simulate noisy acquisition — 15 minutes

Add independent Gaussian noise with standard deviation 0.20 and clip to [0,1]. Explain the consequences of clipping. Keep validation and test noise fixed; generate fresh training noise each epoch. Display clean/noisy pairs. This artificial corruption is a controlled experiment.

## 4. Restore RGB images with a GRU — 50 minutes

Use one **unidirectional**, single-layer GRU with hidden size 128 and batch_first=True, followed by a shared Linear(128,84) layer and sigmoid.

Represent each image as a sequence of 28 rows. Each time step contains 28 pixels × 3 colour channels = **84 features**. Complete the permutation and reshape, process all recurrent outputs, then restore shape (B,3,28,28). Start each image with an independent hidden state.

Calculate parameters manually **before** running the automatic counter:

- Input weights for the three gates: 3hd.
- Recurrent weights: 3h².
- PyTorch's two bias vectors per gate: 6h.
- Shared output projection: 84(h+1).

Report separate totals for the GRU alone, the projection and the complete denoiser. Explain why the sequence length does not multiply the number of weights. Verify your answers against named parameter shapes and sum(p.numel() for p in model.parameters()).

Complete MSE loss and Adam (learning rate 0.001). Train for 25 epochs with noisy input and clean target. Plot training and validation loss. Explain why the last hidden state alone cannot directly provide one output per row, and which context the first reconstructed row can access.

## 5. Evaluate restoration — 25 minutes

Implement per-image MSE and PSNR for pixel range [0,1], then average over test images. Protect logarithms against zero. Compute both metrics for noisy images and restored images, and the relative decrease in mean MSE. Display clean/noisy/restored RGB triplets using identical display settings.

Discuss noise removal, colour fidelity, boundaries and lost fine details. Explain why better average MSE does not imply that every image improves. If time remains, change hidden size to 32, calculate its parameters and compare on validation. Keep the final test set for reporting.

## 6. Conclusion and submission — 10 minutes

Submit the completed student notebook with both manual parameter tables, confusion matrix, loss plots, restoration metrics, image comparisons and written answers. Write 5–10 lines comparing outputs, losses, parameter sharing and limitations of the two models. Restart the kernel and execute all required cells before submission.
