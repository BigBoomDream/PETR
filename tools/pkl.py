import pickle

def print_nested_keys(d, prefix=''):
    """
    递归打印字典中所有层级的key
    d: 字典
    prefix: 用于显示层级的字符串前缀
    """
    if isinstance(d, dict):
        for k, v in d.items():
            print(f"{prefix}{k}")
            if isinstance(v, (dict, list)):
                print_nested_keys(v, prefix + '  ')
    elif isinstance(d, list) and len(d) > 0:
        # 只查看列表第一个元素的结构
        print_nested_keys(d[0], prefix + '  ')

# 读取 pkl 文件
with open('/data/zmz/Nuscenes/nuscenes/mmdet3d_nuscenes_30f_infos_val.pkl', 'rb') as f:
    data = pickle.load(f)

# 打印所有层级的key
print("文件中的数据结构：")
print_nested_keys(data)