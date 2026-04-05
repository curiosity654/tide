# TIDE 扩展到 3D 目标检测的数据与架构重设计建议

本文面向将当前 2D TIDE 扩展到 3D 检测评估场景（KITTI / nuScenes / Waymo）的工程设计。

## 1. 现状与核心限制

当前仓库的主流程是围绕 2D box/mask（COCO 风格）构建：

- `tidecv/data.py`：2D 标注与检测结果容器
- `tidecv/quantify.py`：匹配、误差归因、指标汇总
- `tidecv/ap.py`：AP 计算
- `tidecv/datasets.py`：COCO/LVIS/Pascal/Cityscapes 2D 数据加载器

直接用于 3D 会遇到以下问题：

1. 缺少 3D box 表达（中心点、尺寸、朝向、速度等）
2. 匹配逻辑依赖 2D IoU，无法复用 KITTI / nuScenes / Waymo 的主流指标
3. 误差类型仅覆盖 2D 场景，缺少深度/朝向/速度等 3D 误差
4. 数据集坐标系不统一（camera / lidar / ego / global）

## 2. 目标设计原则

1. **2D 向后兼容**：现有 API 与结果不破坏
2. **插件化评估后端**：2D 与 3D 分离，避免在单一路径中堆条件分支
3. **数据层标准化**：先统一 3D box 语义，再做指标/误差分析
4. **数据集适配器化**：KITTI/nuScenes/Waymo 通过 adapter 接入，不污染核心逻辑

## 3. 推荐重设计（分层）

### 3.1 数据模型层（Data Model）

建议在 `tidecv/data.py` 基础上扩展通用 annotation schema：

- 新增 `task_type`: `2d_det | 2d_seg | 3d_det`
- 新增 3D box 规范字段（推荐 canonical）：
  - `center`: `[x, y, z]`
  - `size`: `[l, w, h]`
  - `yaw`: 绕 z 或 y 轴旋转（需在元数据声明约定）
  - `velocity`（可选）
  - `frame`: `camera | lidar | ego | global`

并提供：

- `add_ground_truth_3d(...)`
- `add_detection_3d(...)`
- 坐标系转换钩子（至少保证同一评估 run 内 frame 一致）

### 3.2 匹配与指标层（Evaluator / Metrics）

建议在 `tidecv/quantify.py` 中抽象 evaluator 接口：

- `BaseEvaluator`
- `Evaluator2D`（复用现有逻辑）
- `Evaluator3D`（新增）

`Evaluator3D` 中按数据集配置选择匹配与指标：

- KITTI：3D AP / BEV AP / AOS（按类别阈值）
- nuScenes：mAP（distance threshold）+ NDS 分项（ATE/ASE/AOE/AVE/AAE）
- Waymo：LEVEL_1/2 + mAPH（按官方定义）

这样可以把“匹配规则 + 指标定义”从通用流程中解耦。

### 3.3 误差归因层（Error Taxonomy）

在 `tidecv/errors/` 新增 3D 误差类型（建议）：

- `DepthError`：距离/深度估计偏差
- `OrientationError`：yaw 偏差
- `ScaleError`：长宽高尺度偏差
- `VelocityError`：速度方向或大小偏差（时序数据集）
- 保留并复用：`ClsError`、`DuplicateError`、`BackgroundError`、`MissedError`

同时保留 TIDE 的“误差修复后 dAP 下降贡献”思想，扩展为 3D 版本。

### 3.4 数据集接入层（Dataset Adapters）

在 `tidecv/datasets.py` 增加独立 loader/adapters（建议命名）：

- `KITTI3D(...)`, `KITTI3DResult(...)`
- `NuScenes3D(...)`, `NuScenes3DResult(...)`
- `Waymo3D(...)`, `Waymo3DResult(...)`

每个 adapter 只做：

1. 读取原始标注/预测
2. 转为统一 annotation schema
3. 声明元信息（坐标系、类别映射、评估协议版本）

### 3.5 可视化与报告层

`tidecv/plotting.py` 建议新增 3D 视图：

- BEV（鸟瞰）误差图
- 深度/朝向/尺度误差直方图
- 数据集特定指标表（KITTI/nuScenes/Waymo）

## 4. 最小可落地实施路径（建议分阶段）

### Phase 1（最小闭环）

1. 数据模型支持 3D box（不改现有 2D 行为）
2. 新增 `Evaluator3D` 骨架与 3D 匹配接口
3. 打通 KITTI 3D 的最小评估闭环（先支持 AP/BEV AP）

### Phase 2（多数据集）

1. 接入 nuScenes（先对齐官方检测评估入口）
2. 接入 Waymo（先对齐官方 metric 导出）
3. 完善 3D 误差分类与汇总报告

### Phase 3（工程化）

1. 增加回归测试与小样本 fixture
2. 统一配置入口（阈值、类别映射、坐标系约定）
3. 优化计算性能（批量匹配、向量化）

## 5. API 草案（示意）

```python
from tidecv import TIDE, datasets

tide = TIDE()

gt = datasets.KITTI3D("/path/to/kitti", split="val")
pred = datasets.KITTI3DResult("/path/to/preds.json")

tide.evaluate_3d(
    gt=gt,
    preds=pred,
    protocol="kitti_3d",
    iou_thresholds={"Car": 0.7, "Pedestrian": 0.5, "Cyclist": 0.5},
)

tide.summarize()
tide.plot()
```

## 6. 与当前仓库兼容策略

1. `evaluate(...)` 保持现状（2D）
2. 新增 `evaluate_3d(...)`，避免破坏已有调用方
3. 3D 相关能力尽量放在新模块（如 `tidecv/eval3d.py`），降低回归风险
4. 文档明确：不同数据集优先对齐官方评估协议，再映射到统一 TIDE 误差视图

---

如果只做“最小改造”，建议先完成 **KITTI 单数据集端到端**，验证架构后再扩展 nuScenes 与 Waymo。
