# WSL_DXG 与 librocdxg 对齐台账

更新时间：2026-04-11  
适用范围：`src/wsl_dxg/*`（仅 `/dev/dxg` 路径），目标 `gfx1100/gfx1201`。  
原则：能对齐就对齐；不能对齐必须给出原因；禁止“找不到就回退”。

## 对齐状态总览

| 项目 | librocdxg 参考 | 当前实现 | 状态 | 说明 |
|---|---|---|---|---|
| 适配器枚举主路径 | `librocdxg/src/wddm/device.cpp:524-542`（仅 `D3DKMTEnumAdapters2`） | `src/wsl_dxg/mod.rs:1607-1610`（仅 `EnumAdapters2`） | 已对齐 | 已移除 `EnumAdapters3` 退避路径。 |
| AMD 设备筛选 | `device.cpp:553-557`（VendorID + Supported） | `src/wsl_dxg/mod.rs:1649-1655`（VendorID + `DxgThunkDeviceInfo::new` 可用性） | 基本对齐 | 都先筛 AMD，再进入后续初始化。 |
| AdapterInfo 私有 blob 解析 | `thunk_proxy` 思路（ADAPTERINFOEX/ATI/PROXY） | `src/wsl_dxg/thunk_proxy.rs:68-79` | 已对齐 | 结构与偏移策略一致。 |
| 引擎序号/HWS/禁用超时掩码 | `thunk_proxy` 掩码 + engine ordinal | `src/wsl_dxg/thunk_proxy.rs:116-184` | 已对齐 | 严格化为 `Result`，不再 `unwrap_or(false)`。 |
| ClockCalibration 计时 | `librocdxg/src/wddm/device.cpp:612-621` | `src/wsl_dxg/mod.rs:1394-1428` | 已对齐 | 使用 `D3DKMTQueryClockCalibration`；并按 `/dev/dxg` 打包 ABI 将 `ClockData` 按 32-bit lane 读取再拼 64-bit，避免自然对齐导致的脏频率。当前 `gfx1201` 实测 `calib.gpu_freq=100000000`（与 adapter info 的 `gpu_counter_frequency` 一致）。 |
| 基准计时返回语义 | librocdxg 以 GPU 时间戳差值作为 kernel 执行时间（CPU 时钟仅辅助） | `src/ignis/gpu_context.rs:382-480` | 已对齐 | `dispatch_batch_profiled_gpu_us` 返回 GPU 时间戳换算值（`gpu_avg_us`）；`wall_avg_us` 仅诊断。换算频率使用 adapter info 的 `gpu_counter_frequency`（与 librocdxg `GPUCounterFrequency()` 一致），`QueryClockCalibration` 保留为诊断链路输出；任何频率异常直接报错，不回退到 wall time。 |
| `amd_signal_t` timestamp 槽位 | librocdxg `amd_signal_t`：`start_ts@+32`、`end_ts@+40` | `src/wsl_dxg/mod.rs` 常量偏移 + `verify_amd_signal_layout_once()` | 已对齐 | 新增运行时一次性校验：`size=64`、`align=64`、`kind/value/start_ts/end_ts` 偏移必须分别为 `0/8/32/40`；并对 `COPY_DATA` 目的地址强制 8 字节对齐，保证 timestamp 写入目标正确。 |
| Kernel dispatch 主 PM4 序列 | `librocdxg/src/wddm/queue.cpp:660-730` | `src/wsl_dxg/mod.rs:3624-3694` | 已对齐 | `start_ts -> barrier(条件) -> acquire -> (gfx11+)scratch+rsrc3 -> dispatch -> barrier -> end_ts -> acquire -> atomic signal -> atomic/read_ptr`。 |
| `COMPUTE_RESOURCE_LIMITS` 连续寄存器顺序 | `librocdxg/src/wddm/cmd_util.cpp:209-217` | `src/wsl_dxg/mod.rs:3650-3660` | 已对齐 | 顺序为 `RESOURCE_LIMITS, SE0, SE1, TMPRING, SE2, SE3`。 |
| RSRC1.PRIV 置位条件 | `cmd_util.cpp:202-205`（`major == 11`） | `src/wsl_dxg/mod.rs:4786-4794`（`major == 11`） | 已对齐 | 已改为仅 gfx11 置位，gfx12 不再强制 patch。 |
| 信号等待策略 | librocdxg 不依赖 read_ptr 伪完成 | `src/wsl_dxg/mod.rs:4448-4471` | 已对齐 | 已删除 `wait_signal` 的 read_ptr fallback。 |
| `wait_idle` 退休语义 | librocdxg 以同步对象/提交链确保真实完成 | `src/wsl_dxg/mod.rs:4452-4464`（`barrier + signal`） | 已对齐（语义层） | 不再直接信任 `read_ptr`；`wait_idle` 改为注入 barrier 包并等待 completion signal，避免 DXG 路径“读指针先行”导致的早返回/后续硬挂。 |

## 未完全对齐项（含原因）

| 项目 | librocdxg 参考 | 当前实现 | 状态 | 原因与结论 |
|---|---|---|---|---|
| `gfx major` 来源 | librocdxg 由 thunk/proxy 体系给出 `major`（非 device-id 硬编码） | `src/wsl_dxg/thunk_proxy.rs:397-409` 通过 device-id 映射 | 未对齐（计划改） | 当前仓库只有 `libthunk_proxy.a` 二进制，缺少完整可复用源码接口。暂用白名单映射且“未知即报错”，无回退。后续若提取到可调用接口，应改为与 librocdxg 同源解析。 |
| 代码对象分配域 | librocdxg 代码对象走 Local VRAM + 显式映射 | `src/wsl_dxg/mod.rs:3129-3139`（`alloc_code -> system`） | 未对齐（有条件） | 在当前 `gfx1201 + /dev/dxg` 机型上，`Local/UserQueue` 分配执行 `D3DKMTLock2` 会返回 `0xC000000D`，导致代码对象无法写入。现阶段保留 system 可见内存作为确定策略；后续若补齐“staging->local 拷贝”链路，再切回 Local。 |
| 通用 VRAM 资源分配域 | librocdxg 默认 Local（仅 system/userptr 才走 system） | `src/wsl_dxg/mod.rs:2474-2489` 将 `flags.vram` 归到 system | 未对齐（有条件） | 当前本项目大量路径要求 `cpu_ptr` 直接可见；为保持现有调用约定，普通 VRAM 仍走 system。后续需引入 staging/copy 后再完全对齐到 Local。 |
| `compute_tmpring_size` 写入 | librocdxg 写队列维护值（`cmd_util.cpp:216`） | `src/wsl_dxg/mod.rs:3657` 固定写 `0` | 未对齐（有条件） | 当前实现明确禁止非零 `private_segment_size`（`src/wsl_dxg/mod.rs:3540-3544`），scratch 队列态尚未接入；在此约束下写 0 与行为一致。若后续启用 scratch，必须改成与 librocdxg 相同的 tmpring 计算与下发。 |
| `COPY_DATA` timestamp 目标地址编码 | librocdxg `CmdUtil::BuildCopyData` 通过 C bitfield 设置 `dst_64b_addr_lo = PtrLow32 >> 3` | `src/wsl_dxg/mod.rs` `copy_gpu_clock_count()` 写原始 `addr_lo`（不预移位） | 未对齐（有实测原因） | 在当前 `gfx1201 + /dev/dxg` 环境，按 `>>3` 预移位会把目标地址错误编码为 `0x7F6B1F28...` 并触发 `ERROR_DMAPAGEFAULT`。改为原始 `addr_lo` 后，PM4 ordinal 与硬件实际解释一致。后续若引入与 librocdxg 同构的 PM4 bitfield 打包层，再评估回归到 `>>3`。 |
| `/dev/dxg` 显式探测前置 | librocdxg 依赖 dxcore/thunk 调用，不先 `open("/dev/dxg")` | `src/wsl_dxg/mod.rs:1590-1605` 先探测且失败即报错 | 有意不对齐 | 本项目目标是“运行前置条件显式依赖 `/dev/dxg`”，因此保留该硬前置检查；同时不做任何继续执行的退避。 |
| 平台原子能力环境变量覆写 | librocdxg 无该运行时覆写开关 | `src/wsl_dxg/mod.rs:185-191, 1359-1361` | 有意不对齐 | 仅用于排障和 A/B，默认仍走 AdapterInfo 真值；并且覆写是显式设置，不是回退逻辑。 |

## 与 docs 经验对齐说明

以下差异或保护与项目已有排障经验一致：

- `docs/WGP_Mode_Page_Fault_根因分析.md`
- `docs/GPU连续运行硬挂防御_实验记录.md`
- `docs/WGP_Mode_RSRC1_CWSR根因分析.md`

核心策略：

1. 不隐藏错误：等待/计时失败直接报错，不用 read_ptr 伪成功。
2. 不做静默回退：未知 device-id、找不到适配器、缺计时能力均直接失败。
3. 可验证差异要落文档：凡与 librocdxg 不一致之处，必须有“原因+后续触发条件”。
4. 频率域要区分：`gpu_counter_frequency`（当前实测 100MHz）是时间戳计数器频率，不等于核心最高时钟。`GFX1201/GCD` 的峰值核心频率（如 2520MHz）只能用于硬件能力讨论，不能直接用于 timestamp tick→time 换算。
5. 计时结果输出规则：严格按 librocdxg 同类链路（GPU timestamp + counter frequency）换算后原样输出；不得因“看起来过高/过低”而抑制、替换或门禁结果。正确性校验信息可并行输出，但不用于屏蔽计时结果。
6. 禁止经验阈值门禁：计时链路中不添加“合理区间/可疑范围”类判断来拦截频率或耗时，除零值/系统调用失败外一律按实测值输出。

## 后续维护规则（强制）

1. 每次改 `src/wsl_dxg/*`，先更新本文件对应行的“状态/原因”。
2. 新增任何 fallback/重试/估计值前，必须先在本文件新增条目并写明必要性；否则不允许合入。
3. 当 `compute_tmpring_size` 或 `major` 解析方案升级后，必须将对应条目状态改为“已对齐”并附代码位置。

## 计时链路验真记录（gfx1201 + /dev/dxg）

最近一次验证（`test_wgp_k64_benchmark`, `log.wgp`）：

1. 频率一致性：`freq_hz=100000000`，且 `calib.gpu_freq=100000000`，两条来源一致。
2. 计数器单调性：`calib[gpu_counter]`、`calib[cpu_counter]` 均严格递增。
3. 计数器比例一致性：相邻 profile 点 `Δgpu_counter / Δcpu_counter ≈ 10.00016`，对应 GPU 计数器 100MHz、CPU 计数器 10MHz 的稳定比例关系。
4. 结论：时间戳链路与频率读取在数值上自洽，属于“已获取到有效时间/频率”的状态；吞吐高低按实测原样输出，不做阈值门禁。
