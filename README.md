# ComfyuiWorkflow

基于 **FLUX.2 [klein] 4B** 的"真人照片 → 日系动漫插画 → 抠图"工作流。

## 文件说明

### workflows/

| 文件 | 说明 |
|---|---|
| `flux2-V3-保存分割-仅插画.json` | **原始工作流**。单条链:FFHQ 人脸 → BiRefNet 抠图 → 贴黑底 → FLUX.2 Klein 图生图转动漫插画 → 提饱和度 → 二次 BiRefNet 抠图。一次跑到底,中断需从头再来。 |
| `flux2-V3-分段可续跑-仅插画.json` | **重构版**。拆成 3 条独立链(① 原图分割 / ② 插画生成 / ③ 二次分割),支持断点续跑、分批处理、区间选择。**依赖下面的自定义节点。** |

### custom_nodes/

| 文件 | 说明 |
|---|---|
| `resume_image_loader.py` | `LoadUnprocessedImagesFromFolder` 节点。重构版工作流必需。 |

## 安装自定义节点

把 `custom_nodes/resume_image_loader.py` 复制到你的 ComfyUI 安装目录下:

```
ComfyUI/custom_nodes/resume_image_loader.py
```

然后**重启 ComfyUI** 一次,让节点注册生效。

## 重构版用法

打开 `flux2-V3-分段可续跑-仅插画.json`,画布上三段分三个彩色组,互相独立:

- **① 原图分割**:读 `input/ffhq/images1024x1024` → BiRefNet 抠图 → 存 `output/cutout/`
- **② 插画生成**:读 `output/cutout/` → 贴黑底 → FLUX 图生图 → 提饱和度 → 存 `output/illustration/`
- **③ 二次分割**:读 `output/illustration/` → BiRefNet 抠图 → 存 `output/final/`

想只跑某一段:把另外两段静默(mute / Bypass)再排队。三段按 ①→②→③ 顺序各排一次队即可跑完整条链。

### 加载器参数(每段头部节点)

| 参数 | 作用 |
|---|---|
| `read_from` | 读 `input/` 还是 `output/`(B、C 段读上一段输出,用 `output`) |
| `input_folder` | 子文件夹 |
| `done_prefix` | 与该段 SaveImage 的 `filename_prefix` 保持一致,用来数已完成 |
| `start_index` | 从排序后第几张开始(选中间区间用) |
| `count` | 跑多少张,0 = 到末尾 |
| `batch_size` | 每次排队跑几张(控内存) |

### 断点续跑

中断后**直接重新排队**当前段即可——加载器按 `done_prefix` 数已有输出,自动跳过已完成的,从第一张没跑完的接着来。全跑完后再排队是静默 no-op,不报错。

### 选区间跑

例如要跑 FFHQ 第 30000~30999 张:该段加载器设 `start_index=30000`、`count=1000`、`batch_size=50`,Queue 面板 Batch count 设 20。

**约定**:换区间时,要么换一个独立的 `done_prefix`(并同步改该段 SaveImage 的 `filename_prefix`),要么清空该段输出文件夹——否则已完成计数会跨区间错位。

## 模型依赖

- `flux-2-klein-4b.safetensors`(UNet)
- `qwen_3_4b.safetensors`(文本编码器)
- `flux2-vae.safetensors`(VAE)
- `birefnet.safetensors`(背景移除)
