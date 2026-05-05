# 赛题说明
## 赛题背景
精准完成表格语义关系提取是赋能数据集成、数据清洗与知识发现等下游任务的关键钥匙。针对真实场景下标定数据获取困难的痛点，研究如何在少样本（Few-Shot）条件下高效实现表格语义关系提取任务，不仅能大幅降低标注成本，更是提升现代数据流水线智能化水平的核心突破口。

## 任务说明
给定一组表格、一个候选关系集合以及目标列对，要求通过表格的数据从候选关系集合中去预测给定目标列对之间的关系类型，表示这两列之间的关系。

- **输入**：目标列对（Subject，Object）、候选关系集合
- **输出**：Subject，Object，预测的关系类型

## 数据集说明
比赛数据集划分为训练集和测试集，训练集共38699条数据，测试集共4068条数据。本次数据集包含563个候选关系，每个表格中包含两列数据。

| 数据集 | 候选关系数量 | 数据总量 |
|--------|--------------|----------|
| 训练集 | 563          | 38699    |
| 测试集 | 563          | 4068     |

## 输入输出样例
### Input
示例表格 table1：

| Subject                | Object               |
|------------------------|----------------------|
| Jobcenter              | Enzo Cormann         |
| The Boy Who Cried Wolf | Gosho Aoyama         |
| Iter Italicum          | Paul Oskar Kristeller|

候选关系集合：`{area, author, brand, date of birth}`
目标列对：`{Subject, Object }`

### Output
| Subject                | Object               | Label  |
|------------------------|----------------------|--------|
| Jobcenter              | Enzo Cormann         | author |
| The Boy Who Cried Wolf | Gosho Aoyama         | author |
| Iter Italicum          | Paul Oskar Kristeller| author |

## 评价方式

我们将采用一种结合少样本重要性和预测准确性的评价方式，具体而言，我们将少样本重要性与关系类型的样本数量进行关联，某关系类型的样本数据越少表示其少样本重要性越高并赋予更高权重。

---

## 少样本重要性权重
少样本重要性权重计算公式如下：

$$
m\_weights = \frac{counts_{max} - counts_{m} + counts_{min} \times 0.1}{counts_{max} + counts_{min} \times 0.1}
$$

其中：
- $counts_{max}$：样本数量最多的关系类型的样本个数
- $counts_{min}$：样本数量最少的关系类型的样本个数
- $counts_{m}$：当前关系类型的样本个数
- $m\_weights$：当前关系类型的少样本重要性权重

为保证每个少样本重要性权重均大于 0，设置约束项 $counts_{min} \times 0.1$ 保证权重有效性。

---

## 准确性计算规则
关系类型 $m$ 的准确性分数计算公式：

$$
m\_score = \frac{m\_correct}{m\_total}
$$

其中：
- $m\_score$：关系类型 $m$ 的准确性分数
- $m\_correct$：关系类型 $m$ 中预测正确的数量
- $m\_total$：关系类型 $m$ 中需要被正确预测的全部数量

---

## 最终分数计算规则
最终分数计算公式：

$$
Score\_final = \frac{\sum_{m} (m\_weights \times m\_score)}{\sum_{m} m\_weights}
$$