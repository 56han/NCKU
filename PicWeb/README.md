# Dazz 攝影作品分享平台

一個使用 ASP.NET MVC 開發的攝影作品分享平台，提供會員上傳、分享照片，並可進行互動交流。
Demo 影片: https://www.youtube.com/watch?v=hgVfMva0Sxo

## 功能特色

### 🔐 會員系統
- 會員註冊與登入
- 個人資料管理
- 個人作品集展示

### 📸 作品管理
- 照片上傳
- 相機型號選擇（可自行新增）
- 作品描述編輯

### 🤝 社群互動
- 按讚功能
- 留言系統
- 作品分類瀏覽

## 使用技術

- ASP.NET MVC 5
- Entity Framework
- Bootstrap 5
- jQuery
- Font Awesome
- Microsoft SQL Server

## 安裝說明

1. 複製專案
```bash
git clone https://github.com/your-username/Dazz-Photo-Sharing-Platform.git
```

2. 建立資料庫
- 執行 DatabaseScript.sql 建立資料庫結構
- 修改 Web.config 中的連線字串

3. 建立必要資料夾
- 在專案根目錄建立 Images 資料夾（用於儲存上傳的圖片）

4. 執行專案
- 使用 Visual Studio 開啟方案
- 建置並執行專案

## 系統需求
- .NET Framework 4.7.2 或以上
- Microsoft SQL Server
- Visual Studio 2019 或以上

## 專案結構
```
PicWeb/
├── Controllers/         # 控制器
├── Models/             # 資料模型
├── Views/              # 視圖檔案
├── Content/            # CSS 樣式
├── Scripts/           # JavaScript 檔案
└── Images/            # 上傳圖片儲存位置
```

## 作者
萬逸涵

- GitHub: @56han
- Email: annie96289@gmail.com
