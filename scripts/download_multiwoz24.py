"""
Script để download MultiWOZ 2.4 dataset từ GitHub
"""

import os
import json
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm


class MultiWOZ24Downloader:
    def __init__(self, data_dir="data/multiwoz24"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # URLs cho MultiWOZ 2.4
        self.base_url = "https://github.com/smartyfh/MultiWOZ2.4/raw/main/data"
        self.urls = {
            "data": f"{self.base_url}/MULTIWOZ2.4.zip",
            "ontology": f"{self.base_url}/ontology.json",
            "valListFile": f"{self.base_url}/valListFile.txt",
            "testListFile": f"{self.base_url}/testListFile.txt"
        }
    
    def download_file(self, url, filename):
        """Download file từ URL"""
        filepath = self.data_dir / filename
        
        if filepath.exists():
            print(f"✓ {filename} đã tồn tại, bỏ qua download")
            return filepath
        
        print(f"Đang download {filename}...")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get total size for progress bar
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress bar
            with open(filepath, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)
            
            print(f"✓ Download {filename} hoàn tất")
            return filepath
            
        except Exception as e:
            print(f"✗ Lỗi khi download {filename}: {e}")
            if filepath.exists():
                filepath.unlink()
            raise
    
    def extract_zip(self, zip_path):
        """Giải nén file zip"""
        print(f"\nĐang giải nén {zip_path.name}...")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get list of files to extract
                file_list = zip_ref.namelist()
                
                # Extract with progress bar
                for file in tqdm(file_list, desc="Extracting"):
                    zip_ref.extract(file, self.data_dir)
            
            print("✓ Giải nén hoàn tất")
            
        except Exception as e:
            print(f"✗ Lỗi khi giải nén: {e}")
            raise
    
    def move_files_from_subfolder(self):
        """Di chuyển files từ subfolder MULTIWOZ2.4 lên thư mục gốc"""
        subfolder = self.data_dir / "MULTIWOZ2.4"
        
        if not subfolder.exists():
            return
        
        print("\nĐang di chuyển files...")
        
        import shutil
        
        # Di chuyển các files cần thiết
        required_files = [
            "data.json",
            "ontology.json",
            "valListFile.json",
            "testListFile.json"
        ]
        
        for filename in required_files:
            src = subfolder / filename
            dst = self.data_dir / filename
            
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
                print(f"✓ Copied {filename}")
    
    def download_all(self):
        """Download tất cả files"""
        print("=" * 70)
        print("BẮT ĐẦU DOWNLOAD MULTIWOZ 2.4 DATASET")
        print("=" * 70)
        
        try:
            # Download data zip
            data_zip = self.download_file(self.urls["data"], "MULTIWOZ2.4.zip")
            
            # Extract zip
            self.extract_zip(data_zip)
            
            # Move files từ subfolder
            self.move_files_from_subfolder()
            
            print("\n" + "=" * 70)
            print("DOWNLOAD HOÀN TẤT!")
            print("=" * 70)
            print(f"Dữ liệu được lưu tại: {self.data_dir.absolute()}")
            
            # Kiểm tra files
            self.verify_download()
            
        except Exception as e:
            print(f"\n✗ Download thất bại: {e}")
            return False
        
        return True
    
    def verify_download(self):
        """Kiểm tra tính toàn vẹn của dữ liệu"""
        print("\n" + "=" * 70)
        print("KIỂM TRA FILES")
        print("=" * 70)
        
        required_files = [
            "data.json",
            "ontology.json", 
            "valListFile.json",
            "testListFile.json"
        ]
        
        all_ok = True
        
        for filename in required_files:
            filepath = self.data_dir / filename
            if filepath.exists():
                size = filepath.stat().st_size / 1024 / 1024
                print(f"✓ {filename:<25} {size:>10.2f} MB")
            else:
                print(f"✗ {filename:<25} KHÔNG TÌM THẤY!")
                all_ok = False
        
        # Load và kiểm tra data.json
        data_file = self.data_dir / "data.json"
        if data_file.exists():
            try:
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"\n✓ data.json hợp lệ: {len(data)} dialogues")
            except Exception as e:
                print(f"\n✗ data.json không hợp lệ: {e}")
                all_ok = False
        
        print("=" * 70)
        
        if all_ok:
            print("✓ Tất cả files đều OK!")
        else:
            print("✗ Một số files có vấn đề!")
        
        return all_ok


def main():
    """Main function"""
    # Đường dẫn tương đối từ scripts/
    downloader = MultiWOZ24Downloader(data_dir="../data/multiwoz24")
    success = downloader.download_all()
    
    if success:
        print("\n🎉 Download thành công! Bạn có thể chạy preprocess_multiwoz24.py tiếp theo.")
    else:
        print("\n❌ Download thất bại! Vui lòng kiểm tra lại.")
        exit(1)


if __name__ == "__main__":
    main()
