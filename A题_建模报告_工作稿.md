## 2026 MCM A 题：智能手机电池耗电建模（工作稿）

> 本文档为**边写边做**的工作稿：先给出可运行、可解释的基准模型与基准结论，再通过“修正项/修正方程”的方式逐步补丁迭代。  
> **主数据集**：`Database/` 文件夹内的设备日志（以设备 IMEI 命名的 CSV）与 `Activation_Date_Phone.xlsx`。  
> 题面来源：`2026_MCM_Problem_A.pdf`（本地：`C:\Regular\2026MCM\2026_MCM_Problem_A.pdf`）。

### 1 引言

#### 1.1 问题重述（Problem Restatement）

Smartphone battery life can vary dramatically from day to day. Beyond the total duration of use, battery drain depends on a combination of screen activity, processor workload, network connectivity (Wi‑Fi vs. cellular), background applications, and environmental factors such as temperature. Long‑term battery aging further alters the effective capacity and discharge behavior.

The objective of this problem is to develop a **continuous‑time** mathematical model for a lithium‑ion smartphone battery that returns the **state of charge** \(SOC(t)\) over time under realistic usage conditions, and to use the model to produce actionable, testable predictions. After reviewing the problem statement, we formulate the task into the following sub‑problems:

- **MODEL**:  
  Construct a continuous‑time equation (or system of equations) that describes battery state evolution \(SOC(t)\) using physically interpretable mechanisms (e.g., energy balance / equivalent‑circuit reasoning), and define all state variables and parameters.

- **PREDICT**:  
  Use the model to estimate **time‑to‑empty** under different initial charge levels and usage scenarios; compare predictions against observed or plausible behavior.

- **DIAGNOSE**:  
  Explain why different scenarios produce different drain rates, and identify the dominant drivers of rapid battery depletion.

- **SENSITIVITY & UNCERTAINTY**:  
  Evaluate how predictions change under variations in assumptions, parameter values, and fluctuations in usage patterns; quantify uncertainty and identify failure modes.

- **RECOMMEND**:  
  Translate model insights into practical guidance for users and potential power‑management strategies for an operating system; discuss how battery aging affects outcomes and how the framework can generalize to other portable devices.

To support parameter estimation and validation, we will use the provided primary dataset `MCM_2026_A/Database/`, which contains per‑device, seconds‑level logs (timestamped battery and context variables such as battery percentage, temperature, voltage, current, network type, screen state, charging state, and foreground app). The accompanying file `Activation_Date_Phone.xlsx` provides device activation dates that can be used to stratify or proxy battery aging.

