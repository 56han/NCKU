# Linux Kernel Performance Analysis

## Table of Contents
- [div64.c](#div64c)
  - [實驗目的](#實驗目的)
  - [實驗步驟](#實驗步驟)
  - [效能評估](#效能評估)
  - [實驗結果分析](#實驗結果分析)
- [memchr.c](#memchrc)
  - [實驗設計](#實驗設計)
  - [分析結果](#分析結果)
  - [x86 vs memchr_opt](#x86-vs-memchr_opt)

---

## div64.c

### **實驗目的**
- 在 Linux 核心原始程式碼找到 `div64.c` 相關的程式碼（`do_div()`）。
    - Linux 核心原始程式碼：[include/asm-generic/div64.h](https://github.com/torvalds/linux/blob/master/include/asm-generic/div64.h)
- 探討 `do_div()` 巨集在不同除數情境下的表現，評估其效能與限制。

### **實驗步驟**
#### **步驟 1：撰寫測試模組**
- 定義多個除法測試情境，包括 **常數除數** 與 **變數除數**。
- 記錄 `do_div()` 的 **執行時間** 和 **結果**。

#### **步驟 2：編譯與執行**
```shell
# 編譯 Linux 核心模組
make

# 載入模組
sudo insmod div_test.ko

# 觀察測試結果
sudo dmesg | grep "do_div test"
```

---

### **效能評估**
| 測試情境 | 平均執行時間 (ns) | 效能提升 |
|----------|----------------|----------|
| 變數除數 | 33 ns         | -        |
| 常數除數 | **25.2 ns**   | **23.53%** |

---

### **實驗結果分析**
#### **常數除數 vs 變數除數**
- **常數除數**：
  - 2<sup>n</sup> 的除數可用 **右移** 取代除法，提高效率。
  - 編譯期可進行優化，減少運行時計算負擔。
- **變數除數**：
  - **無法進行位運算優化**，需完整執行 64-bit 除法。
  - **效能較低**，因為 CPU 需要完整執行除法運算。

---

## memchr.c

### **實驗設計**
1. **字串長度變化**：測試 1KB、10KB、100KB、1MB、10MB 等長度。
2. **特定模式變化**：在字串 **開頭、中間、結尾** 測試 `memchr()`。
3. **多次運行**：減少隨機誤差，每個測試執行多次取平均值。
4. **記錄時間**：使用高解析度計時器 `clock_gettime` 測量時間。

---

### **分析結果**
```shell
Length: 1024, Position: 0, memchr: 0.000000, memchr_opt: 0.000000
Length: 1024, Position: 512, memchr: 0.000000, memchr_opt: 0.000000
Length: 1024, Position: 1023, memchr: 0.000000, memchr_opt: 0.000000
Length: 1024, Position: 1025, memchr: 0.000000, memchr_opt: 0.000000

Length: 10240, Position: 0, memchr: 0.000000, memchr_opt: 0.000000
Length: 10240, Position: 5120, memchr: 0.000000, memchr_opt: 0.000001
Length: 10240, Position: 10239, memchr: 0.000000, memchr_opt: 0.000002
Length: 10240, Position: 10241, memchr: 0.000000, memchr_opt: 0.000002

Length: 102400, Position: 0, memchr: 0.000000, memchr_opt: 0.000000
Length: 102400, Position: 51200, memchr: 0.000000, memchr_opt: 0.000016
Length: 102400, Position: 102399, memchr: 0.000000, memchr_opt: 0.000027
Length: 102400, Position: 102401, memchr: 0.000000, memchr_opt: 0.000049

Length: 1048576, Position: 0, memchr: 0.000000, memchr_opt: 0.000000
Length: 1048576, Position: 524288, memchr: 0.000000, memchr_opt: 0.000124
Length: 1048576, Position: 1048575, memchr: 0.000000, memchr_opt: 0.000277
Length: 1048576, Position: 1048577, memchr: 0.000000, memchr_opt: 0.000242

Length: 10485760, Position: 0, memchr: 0.000000, memchr_opt: 0.000000
Length: 10485760, Position: 5242880, memchr: 0.000000, memchr_opt: 0.001668
Length: 10485760, Position: 10485759, memchr: 0.000000, memchr_opt: 0.003597
Length: 10485760, Position: 10485761, memchr: 0.000000, memchr_opt: 0.002807
```

#### **影響效能的因素**
- **函數呼叫開銷**：`memchr_opt` 有額外的對齊檢查，影響小資料量效能。
- **對齊檢查**：小資料塊時，對齊、優化的開銷可能超過潛在收益。
- **分支預測失敗**：`memchr_opt` 含多個條件分支，影響 CPU 預測效能。
- **記憶體存取模式**：大資料塊時 `memchr_opt` 才能展現優勢。

---

## x86 vs `memchr_opt`
Linux 核心原始程式碼找出 x86 對應的最佳化實作：[arch/x86/lib/string_32.c](https://github.com/torvalds/linux/blob/master/arch/x86/lib/string_32.c)

### **x86 最佳化策略**
- **利用 x86 指令集**：`repne scasb` 進行快速字節掃描。
- **內嵌 Assembly 優化**：減少函數呼叫開銷，提高效能。

### **memchr_opt 最佳化策略**
- **記憶體對齊檢查**：提高效能但增加初始開銷。
- **多字節處理**：批量處理資料，提高速度。
- **動態回退策略**：無法使用 SIMD 時回退到單字節掃描。

---

### **結論**
| 實作方式 | 優勢 | 劣勢 |
|----------|----------------|----------------|
| x86 `repne scasb` | 高度優化、指令級別優化 | 受限於 x86 架構 |
| `memchr_opt` | 跨平台適用、適合大資料塊 | 小資料塊效能較低 |

---

## 📌 參考連結
- [Linux Kernel Source - div64.h](https://github.com/torvalds/linux/blob/master/include/asm-generic/div64.h)
- [Linux Kernel Source - x86 string operations](https://github.com/torvalds/linux/blob/master/arch/x86/lib/string_32.c)
