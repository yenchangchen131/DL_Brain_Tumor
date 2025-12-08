import json
import os

def clean_dataset(json_path, image_ids_to_remove, output_path):
    """
    清理資料集,移除指定的 image IDs
    
    參數:
        json_path: 原始 JSON 檔案路徑
        image_ids_to_remove: 要移除的 image ID 集合
        output_path: 輸出的清理後 JSON 檔案路徑
    """
    print(f"\n{'='*80}")
    print(f"處理檔案: {json_path}")
    print(f"{'='*80}")
    
    try:
        # 讀取 JSON 檔案
        print(f"正在載入 JSON 檔案...")
        if not os.path.exists(json_path):
            print(f"錯誤:找不到 JSON 檔案 '{json_path}'。")
            return False
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print("JSON 檔案載入成功。")
        
        # 記錄原始數量
        original_image_count = len(data.get('images', []))
        original_annotation_count = len(data.get('annotations', []))
        print(f"原始圖片數量: {original_image_count}")
        print(f"原始標註數量: {original_annotation_count}")
        
        # 移除指定的 Image IDs
        print(f"\n正在移除 Image IDs {list(image_ids_to_remove)}...")
        
        removed_image_count_total = 0
        removed_annotation_count_total = 0
        
        # 過濾 images 列表
        if 'images' in data:
            original_images = data['images']
            # 保留 ID 不在列表中的圖片
            data['images'] = [img for img in original_images if img.get('id') not in image_ids_to_remove]
            removed_image_count_total = original_image_count - len(data['images'])
            print(f"- 已從 'images' 移除 {removed_image_count_total} 項。")
        
        # 過濾 annotations 列表
        if 'annotations' in data:
            original_annotations = data['annotations']
            # 保留 image_id 不在列表中的標註
            data['annotations'] = [ann for ann in original_annotations if ann.get('image_id') not in image_ids_to_remove]
            removed_annotation_count_total = original_annotation_count - len(data['annotations'])
            print(f"- 已從 'annotations' 移除 {removed_annotation_count_total} 項。")
        
        # 記錄處理後數量
        final_image_count = len(data.get('images', []))
        final_annotation_count = len(data.get('annotations', []))
        print(f"\n處理後圖片數量: {final_image_count}")
        print(f"處理後標註數量: {final_annotation_count}")
        
        # 儲存清理後的資料
        print(f"\n正在儲存清理後的資料到 {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"已將清理後的資料儲存至 {output_path}")
        
        return True
        
    except FileNotFoundError:
        print(f"\n錯誤:找不到 '{json_path}'。")
        return False
    except Exception as e:
        print(f"\n發生未預期的錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False

# ===== 主程式 =====
print("="*80)
print("資料清理程式 - 對驗證集和測試集套用相同的清理")
print("="*80)

# 定義要移除的 ID 列表 (與訓練集相同)
# 這包含上面發現的 1380 和問題的 1005
image_ids_to_remove = {1380, 1005}
print(f"\n將移除的 Image IDs: {list(image_ids_to_remove)}")

# 處理驗證集
valid_json_path = "./valid/_annotations.coco.json"
valid_output_path = "./valid/_annotations.coco.cleaned.json"
valid_success = clean_dataset(valid_json_path, image_ids_to_remove, valid_output_path)

# 處理測試集
test_json_path = "./test/_annotations.coco.json"
test_output_path = "./test/_annotations.coco.cleaned.json"
test_success = clean_dataset(test_json_path, image_ids_to_remove, test_output_path)

# 總結
print("\n" + "="*80)
print("總結")
print("="*80)
print(f"驗證集清理: {'成功' if valid_success else '失敗'}")
print(f"測試集清理: {'成功' if test_success else '失敗'}")
print("\n注意:這些變更已儲存到新的檔案(_annotations.coco.cleaned.json)")
print("如果要使用清理後的資料,請在後續程式碼中使用這些新檔案。")
print("\n--- 程式執行完畢 ---")
