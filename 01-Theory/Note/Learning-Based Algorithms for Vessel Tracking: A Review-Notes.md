### 1.Introduction

**血管追踪问题（vessel tracking）的三大应用：**关键点探测（key-point detection），中心线提取（centerline extraction），血管分割（vascular segmentation）



**血管追踪的问题因素**被分为2类：**A. 与血管形态有关**（小血管、血管分叉匹配、弯曲程度、极细血管），**B. 与图像特点有关**（低对比度、噪声、伪影、错位、相邻目标交融）



**不断推动算法进步的一些问题：**定位血管的关键点，血管结构的模式识别，基于对血管外形的先验知识和内在特征建立假设并设计模型



`Suri et al.(2002) `
 `A Review on MR Vascular Image Preocessing-Skeleton versus Nonskeleton Approches `

**Skeleton-based/Indirect techniques：**根据血管的2D切片计算血管骨架（skeleton），以分割/重建血管

**Nonskeleton-based/direct techniques：**直接在3D场景下分割/重建血管



`Kirbas and Francis(2004)`

`A Review of Vessel Extraction Techniques and Algorithms`

综述：检测**类血管特征**的方法，如神经血管和管状结构



`Rudyanto et al.(2014)`

`Comparing algorithms for automated vessel segmen- tation in computed tomography scans of the lung: the VESSEL12 study.`

综述：**肺部**CT血管分割，可能与实验室课题相关



` Moreno and Smedby(2015)` 

`Gradient-basedenhancement of tubular structures in medical images`

血管增强方法的综述



`Kerrien et al.(2017)`

`Blood vessel modeling for interactive simulation of interventional neuroradiology procedures`

血管分割的模型方法

---







### 2. Vessel tracking using conventional machine learning 传统学习方法

传统的学习算法：手工制作的特征，分类，数据模型



**Table1～3:** 用于分割**视网膜血管**/**冠状血管**/**其他血管**的传统学习算法

**Table4:** 将现有的传统学习方法归入3个子类中（Hand-crafted features, Classifications, Statistical models）



#### 2.1 Hand-crafted features 手工设计特征

**手工设计特征**，依照人类视觉的特点对什么样的特征敏感，什么样的特征不敏感来提取图像中有区分能力的特征，因此提取出来的特征每一维往往都有具体的**物理含义**。



**与idea相关：**

1. Bogunovic (2012)：提取了**血管分岔特征向量** 
2. Mehmet (2016)：**Hessain矩阵**对**管状/非管状的区分**有效



**Hand-crafted features的应用：**

1. 实现“学习内核（learning-based kernel）”
2. 基于学习的血管滤波器



#### 2.2 Classifications

**无监督算法：**以聚类为主，**k-means**和**fuzzy C-means**

**有监督算法：**支持向量机SVM，Boosting-Based methods，随机森林



#### 2.3 Statistical models



---







### 3. Vessel tracking based on deep learning

#### 3.1 Frameworks of vessel tracking

通过由**统一框架(Unified vessel-tracking methods)**或**两步走的方案(Two-step scheme)**可以得到**分层特征**，并进一步可以得到血管追踪结果。



- ==**统一框架(Unified vessel-tracking methods)**==：将**特征提取**和**像素分类**整合进一个神经网络中

**像素分类的实现**：将CNN的输出层与全连接层相连接。

**双神经元**输出层——典型，用于从图像中分割出血管区域。**多神经元**输出层——用于同时分割血管和其他结构，相当于多任务学习



**CNN做像素分类的缺陷：**用于分类的全连接层所接受的输入范围过大，导致输出图片过于粗糙。

**改进**：改为输出尺寸相同的label maps。每个像素的label由该像素周边的image patches共同影响。



- ==**两步走的方案(Two-step scheme)**==：先使用**CNN进行特征学习**，提取血管特征，在此基础之上，再使用**传统方法**进行血管追踪

**Step1-特征学习：**使用CNN，将输入的图片映射到**中间表示（intermediate representation）**中。通常的映射方式有：概率映射、几何映射、其他映射等。

**Step2-血管追踪：**将**传统的血管追踪方法**应用到Step1中所学习到的中间表示中，最终得到血管追踪结果。提到的传统方法：对概率映射做阈值滤波、RFs、voting scheme、正则步行算法等。











### 论文阅读

`Li et al.(2016) `

`A Cross-Modality Learning Approach for Vessel Segmentation in Retinal Images`

**图像中的血管宽度**：**1～20像素**之间，受血管的实际解剖学宽度和图像分辨率共同影响



**血管分割的难点**：血管交叉、分叉、中心线反射；血管的病变和一些渗出物会导致难度增加



**血管分割的无监督方法**：匹配滤波(matched filtering)，血管追踪（vessel tracking），基于模型的方法（model-based approaches）



















