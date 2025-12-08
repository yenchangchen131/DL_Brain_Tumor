# ============================================================
# 資料清理 - 驗證集和測試集
# ============================================================
# 此 cell 對驗證集和測試集套用與訓練集相同的資料清理操作
# 移除有問題的圖片 ID: 1380 (無標註) 和 1005 (多重標註)

import json
import os

def clean_dataset(json_path, image_ids_to_remove, output_path):
    """
    清理資料集,移除指定的 image IDs
    
    參數:
        json_path: 原始 JSON 檔案路徑
        image_ids_to_remove: 要移除的 image ID 集合
        output_path: 輸出的清理後 JSON 檔案路徑
    
    返回:
        bool: 成功返回 True,失敗返回 False
    """
    print(f"\n{'='*60}")
    print(f"處理: {json_path}")
    print(f"{'='*60}")
    
    try:
        # 讀取 JSON 檔案
        if not os.path.exists(json_path):
            print(f"警告: 找不到檔案 '{json_path}',跳過處理。")
            return False
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 記錄原始數量
        original_image_count = len(data.get('images', []))
        original_annotation_count = len(data.get('annotations', []))
        print(f"原始: {original_image_count} 張圖片, {original_annotation_count} 個標註")
        
        # 過濾 images 列表
        if 'images' in data:
            data['images'] = [img for img in data['images'] 
                            if img.get('id') not in image_ids_to_remove]
            removed_images = original_image_count - len(data['images'])
        
        # 過濾 annotations 列表
        if 'annotations' in data:
            data['annotations'] = [ann for ann in data['annotations'] 
                                 if ann.get('image_id') not in image_ids_to_remove]
            removed_annotations = original_annotation_count - len(data['annotations'])
        
        # 記錄處理後數量
        final_image_count = len(data.get('images', []))
        final_annotation_count = len(data.get('annotations', []))
        
        print(f"移除: {removed_images} 張圖片, {removed_annotations} 個標註")
        print(f"剩餘: {final_image_count} 張圖片, {final_annotation_count} 個標註")
        
        # 儲存清理後的資料
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"✓ 已儲存至: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"錯誤: {e}")
        return False

# ===== 主程式 =====
print("="*60)
print("開始清理驗證集和測試集")
print("="*60)

# 定義要移除的 ID (與訓練集相同)
image_ids_to_remove = {1380, 1005}
print(f"\n要移除的 Image IDs: {list(image_ids_to_remove)}")

# 處理驗證集
valid_success = clean_dataset(
    json_path="./valid/_annotations.coco.json",
    image_ids_to_remove=image_ids_to_remove,
    output_path="./valid/_annotations.coco.cleaned.json"
)

# 處理測試集
test_success = clean_dataset(
    json_path="./test/_annotations.coco.json",
    image_ids_to_remove=image_ids_to_remove,
    output_path="./test/_annotations.coco.cleaned.json"
)

# 總結
print("\n" + "="*60)
print("清理完成!")
print("="*60)
print(f"驗證集: {'✓ 成功' if valid_success else '✗ 失敗'}")
print(f"測試集: {'✓ 成功' if test_success else '✗ 失敗'}")
print("\n注意:")
print("- 清理後的檔案已儲存為 _annotations.coco.cleaned.json")
print("- 如需使用清理後的資料,請載入這些新檔案")
print("="*60)
