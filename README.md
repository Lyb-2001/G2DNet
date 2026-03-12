This is the official repository for our paper, "Generative-to-Discriminative: Uncertainty-Aware Distillation for RGB-T Snowy Urban Scene Parsing". 

🔥 **Highlights:** We pioneer a novel Generative-to-Discriminative (G2D) framework that successfully resolves the trade-off between generative uncertainty modeling and real-time inference. Our exceptionally lightweight student network, **G2DNet-S***, achieves state-of-the-art **83.0% mIoU** while requiring only **5.76 M parameters** and **15.62 G FLOPs**, running at robust **24.76 FPS**.

## Architecture and Details

### 1. Overall G2D Framework & UAD Strategy
<img width="586" height="392" alt="kd" src="https://github.com/user-attachments/assets/e3a469d6-56e8-4bcb-94a5-ab7efc4fa682" />

### 2. Generative Teacher (G2DNet-T)
<img width="767" height="414" alt="teacher" src="https://github.com/user-attachments/assets/c7f167bc-839f-4195-993c-5fa3c123f0ff" />

### 3. Real-Time Student (G2DNet-S) with MSA & FDA
<img width="601" height="327" alt="student" src="https://github.com/user-attachments/assets/8a38e7d0-a451-4dd4-adf5-d1eeb553454f" />

## Result
<img width="1505" height="696" alt="image" src="https://github.com/user-attachments/assets/afabd627-a500-4858-80b2-71d32780f5b3" />

### Quantitative Results (Efficiency vs. Accuracy)
<img width="785" height="667" alt="image" src="https://github.com/user-attachments/assets/e4fbfef1-19c4-41f3-a797-9576923e30bf" />

### Qualitative Visualization
<img width="1523" height="894" alt="image" src="https://github.com/user-attachments/assets/dba71ddc-da9a-4502-bc27-9551c3ccb654" />

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
