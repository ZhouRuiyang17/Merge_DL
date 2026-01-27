# Merge_DL
- 项目流
    0. 检查QPE ACC1H GAUGE_GRID
    1. 生成数据集
    2. 训练和测试
    3. 点的匹配
    4. 评估
- 工作流：
    1. 修改模型model
    2. 修改对应的损失函数和训练过程DLtools
    3. train
    4. test
    5. 把数据取出来point2radar
    6. 评估eval

# 数据集生成
generate_dataset_and_index
- 用npz压缩，8小时有700mb。不对，700是被打断的，实际写完应该有1.05G。
- 用npy存储，写入很快，但是空间飙升至4.98G
- 20260116重新生成数据集，忘记考虑rain_ratio有无效值的情况，还忘记考虑gauge中的无效值是99999，归一化后是499.995，对loss产生严重影响

generate_dataset_and_index_memmap❌
- 2019有48G，吓哭了
- 2026.1.26, 删除

# 训练
### ver1
加载数据集用时: 0.62 秒
划分数据集用时: 0.00 秒
创建数据加载器用时: 0.00 秒
- 2026.1.16和18
Model: base=32, depth=4, n_res=1, norm="nonorm", act="relu"
Loss: MSE Loss
Optimizer: Adam, lr=1e-4, (weight_decay=1e-4)
