import os
import json


# 1. 基本设置
json_list_to_merge = ['seed42_data_6000.json', 
                      'seed8888_data_6000.json']

json_dir = "/media/sdb/liutao/datasets/rm_images/json"
json_list_to_merge = [os.path.join(json_dir, json_one) for json_one in json_list_to_merge]

# 2. 合并

def merge_json_list(json_list_to_merge, output_json):
    merged_data = []

    #  init merged json
    with open(json_list_to_merge[0], 'r') as f:
            data = json.load(f)
   
    for item in data:
        merged_item = {}
        merged_item["ids"] = item["ids"]
        merged_item["text"] = item["text"]
        merged_item["generations"] = item["generations"]

        merged_data.append(merged_item)

    
    # 添加 genenrations
    for _, json_file in enumerate(json_list_to_merge[1:]):
        print(json_file)
        with open(json_file, 'r') as f:
            data = json.load(f)
        for merged_item, item in zip(merged_data, data):      
            generations = item["generations"]
            merged_item["generations"].extend(generations)

    
    # 写入
    cnt = len(merged_data)
    save_json_path = os.path.join(json_dir, f"{cnt}_{output_json}")
    with open(save_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(merged_data, json_file, ensure_ascii=False, indent=4)



if __name__ == "__main__":
    merge_json_list(json_list_to_merge, "merged.json")

