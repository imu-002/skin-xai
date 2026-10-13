# Skin lesion classification with explainable AI

Master's research project (Kyungdong University, M.S. Artificial Intelligence), work in progress.

Three CNNs are trained on HAM10000 to classify dermoscopy images into seven classes. Grad-CAM and LIME explanations are then compared for how well they match the lesion and how faithful they are to the model.

## What it does

- **Data:** HAM10000 (10,015 images, 7 classes) with the official lesion segmentation masks, downloaded from the Harvard Dataverse record.
- **Splits:** grouped by `lesion_id`, so photos of the same lesion never appear in both train and test, and stratified by class. Three seeds.
- **Preprocessing:** hair removal (black-hat filter and inpainting), resized to 224 x 224.
- **Models:** ResNet-50, MobileNetV3-Large, EfficientNet-Lite0 (ImageNet pretrained, via `timm`). Class-weighted cross-entropy, AdamW, cosine schedule, early stopping on validation macro-F1.
- **Metrics:** accuracy, balanced accuracy, macro and weighted F1, macro AUROC, per-class recall.
- **Melanoma threshold:** besides argmax, a decision threshold for melanoma is chosen on the validation set (best balanced accuracy, and at least 80% melanoma recall) and applied once to the test set.
- **Explanations:** Grad-CAM and LIME on test images from every class, scored with
  - IoU with the lesion mask (does the map point at the lesion?)
  - deletion and insertion AUC (does removing the highlighted pixels change the prediction?)
  - a smooth random map as a baseline that any real explanation should beat
- **Efficiency:** parameters, model size, GMACs and CPU latency per model.

## Running it

Open `Skin_XAI_One_Click.ipynb` in Colab, choose a T4 GPU and run all. The notebook downloads the data, runs the tests, trains every model and seed, computes the explanations and writes the result tables. Results are saved to Google Drive so an interrupted run can resume.
