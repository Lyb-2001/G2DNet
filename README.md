<img width="507" height="273" alt="image" src="https://github.com/user-attachments/assets/fe9cc84f-9b75-4af5-8fb2-a48ebd9da530" /># G2DNet: Generative-to-Discriminative Framework for Real-Time RGB-T Scene Parsing

This is the official repository for our paper, "[Your Paper Title Here]". 

🔥 **Highlights:** We pioneer a novel Generative-to-Discriminative (G2D) framework that successfully resolves the trade-off between generative uncertainty modeling and real-time inference. Our exceptionally lightweight student network, **G2DNet-S***, achieves state-of-the-art **83.0% mIoU** while requiring only **5.76 M parameters** and **15.62 G FLOPs**, running at robust **24.76 FPS**.

## Architecture and Details

### 1. Overall G2D Framework & UAD Strategy
<img width="1766" height="945" alt="G2D_framework" src="[Paste your framework image link here, e.g., https://github.com/.../img1.png]" />

### 2. Generative Teacher (G2DNet-T)
<img width="1766" height="824" alt="G2DNet_Teacher" src="[Paste your Teacher/DINOv3/CUA image link here]" />

### 3. Real-Time Student (G2DNet-S) with MSA & FDA
<img width="1766" height="910" alt="G2DNet_Student" src="[Paste your Student network image link here]" />

## Result

### Quantitative Results (Efficiency vs. Accuracy)
<img width="1766" height="738" alt="Quantitative_Results" src="[Paste your table/chart image link here]" />

### Qualitative Visualization
<img width="1766" height="1050" alt="Qualitative_Visualization" src="[Paste your visualization comparison image link here]" />

## Weights and Visualization Results

You can download the pre-trained weights of G2DNet-S* and the visualization results from the following links:

* **Baidu Netdisk:** [Link to your Baidu Drive]([Insert Link Here]) (Extraction code: `[code]`)
* **Google Drive (Optional):** [Link to your Google Drive]([Insert Link Here])

## Base Framework and Dataset

The baseline framework and the datasets (e.g., [Insert dataset name, e.g., MFNet / PST900]) used in our work can be obtained from the following repository/links:

* [Link to the Dataset Repository]([Insert Link Here])

## Code Release

The complete source code, including the training scripts for the Flow Matching Teacher (G2DNet-T), the Uncertainty-Aware Distillation (UAD), and the Student network (G2DNet-S), will be released upon the acceptance of our paper. Stay tuned!

## Contact

Please drop me an email for any problems or discussion: [Your Email Address, e.g., yibenli2001@163.com].
