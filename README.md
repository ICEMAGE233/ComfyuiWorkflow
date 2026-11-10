# ComfyuiWorkflow

基于 **FLUX.2** 的多风格贴纸数据生产工作流 + 批量生成/上传脚本。

## 工作流 (`workflows/`)

| 文件 | 风格 | 模型 |
|---|---|---|
| `flux2-V3-保存分割-仅插画.json` | 日系动漫插画 | flux-2-klein-4b |
| `flux2-V3-分段可续跑-仅插画.json` | 插画(三段式,断点续跑) | flux-2-klein-4b |
| `flux2-V3-保存分割-仅膨胀.json` | 充气气球 | flux-2-klein-4b |
| `anime_style_seg-3D动漫.json` | 3D 动漫渲染 | flux-2-klein-4b + z-image-turbo |
| `flux2-klein-base-4b-像素风.json` | FLUX 像素艺术 | flux-2-klein-base-4b-fp8 |
| `pixel_postprocess.json` | 像素化后处理 | PixelArtDetector |

## 自定义节点 (`custom_nodes/`)

`resume_image_loader.py` — 三个节点:
- `LoadUnprocessedImagesFromFolder`:按文件名编号解析、断点续跑、分批、区间选择
- `SaveImageIndexed`:按编号存储 `output/{prefix}/{idx:05d}.png`
- `UploadToModelScope`:异步上传到 ModelScope 数据集

## 脚本 (`scripts/`)

| 脚本 | 作用 |
|---|---|
| `_runrange_local.py` | 插画风格生成(复用 cutout) |
| `_runrange_inflated.py` | 膨胀风格生成 |
| `_runrange_anime.py` | 3D 动漫生成 |
| `_runrange_fluxpixel.py` | FLUX 像素艺术生成(base-4b) |
| `_runrange_pixel.py` | 像素化后处理(无 FLUX) |
| `_partwatcher_*.py` | 对应的 watcher(逐 Part 上传) |
| `_batchupload.py` | 全量批量上传 |
| `_copy_to_goat.py` / `_copy_inflated_to_goat.py` | 复制到 GoatGroup 仓库 |

## 文档 (`docs/`)

`工作总结.md` + `images/pipeline.svg` — 完整流程说明。

## 模型依赖

- `flux-2-klein-4b.safetensors` / `flux-2-klein-base-4b-fp8.safetensors`
- `qwen_3_4b.safetensors`
- `flux2-vae.safetensors` / `full_encoder_small_decoder.safetensors`
- `birefnet.safeteters`
- `z_image_turbo_bf16.safetensors`(3D 动漫工作流)
