### 生词

**Abstract：**yield 产生，hierarchies 层次结构，the state-of-the-art 最先进的方法，spatially 空间上，dense 密集的，contemporary 同时代的/现代的，fine-tune 微调，semantic 语义的，coarse 粗的，fine 细的

**1.Introduction：**drive advances in... 推动...进步，make progress 取得进展，correspondence 通信，enclosing 封闭的，shortcomings 缺点，asymptotically 渐进的，preclude 排除，interpret 解释，inherent 内在的，tension 压力，resolve 解决，tradeoff 权衡，

---



### 专业术语

**pixel-wise** 像素级别的（类似的：image-wise 图像级别的，patch-size 补丁级别的）

**end-to-end** 端到端：指使用者输入原始材料后，可以直接得到可用的结果，而不用去关心中间的复杂过程（img2img，img2txt，etc.）

---





### 1. Introduction

FCN是**pixel-wise**的，但此前使用CNN做语义分割的方法都是patch-wise的，即根据周围的像素来预测每个像素的类别。



FCN使用了**预训练模型**：通过将优秀的分类网络修改为全卷积网络，并在其预训练参数中进行微调，可以将很成功的分类网络迁移到分割任务中来（相当于密集分类任务，即dense prediction）。此前的分割模型都没有使用过预训练方法，而且都用的是小型的卷积网络。



**语义分割的矛盾**：deep, coarse, semantic info V.S. shallow, fine, appearance info

FCN的解决方案：**跳跃结构（skip architecture）**



### 2. Related work















