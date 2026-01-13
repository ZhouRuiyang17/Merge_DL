# Merge_DL

# 数据集生成
generate_dataset_and_index
- 用npz压缩，8小时有700mb。不对，700是被打断的，实际写完应该有1.05G。
- 用npy存储，写入很快，但是空间飙升至4.98G

generate_dataset_and_index_memmap❌
- 2019有48G，吓哭了