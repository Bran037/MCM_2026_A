## 2026 MCM A 题：智能手机电池耗电建模

### 1 引言（Introduction）

#### 1.1 背景介绍

智能手机已成为现代生活的高频基础设施，但其续航表现往往难以预测：同一部手机在不同日期可能出现显著不同的耗电速度。除“使用时长”外，屏幕亮度与点亮状态、处理器负载、网络连接（Wi‑Fi/蜂窝）、后台应用活动、定位与传感器，以及环境温度等因素都会改变能耗水平。同时，电池的老化与历史充电方式会使有效容量与放电特性随时间变化。

（后续内容留白）

#### 1.2 问题重述（Problem Restatement）

本题要求建立一个**连续时间**的数学模型，以描述锂离子电池的**电量状态** \(SOC(t)\) 随时间的演化，并据此在现实使用条件下产生可验证的预测与可操作的建议。综合题意，可将任务表述为以下子问题：

- **建模（MODEL）**：  
  构建连续时间方程（或方程组）来刻画 \(SOC(t)\) 的变化规律；模型应具有可解释的物理/机理基础（如能量守恒或等效电路思想），并清晰定义所有状态变量与参数。

- **预测（PREDICT）**：  
  在不同初始电量与使用场景下，估计“距离耗尽时间”（time‑to‑empty），并与观测或合理行为进行对比验证。

- **诊断（DIAGNOSE）**：  
  解释不同场景为何产生不同耗电速度，识别导致快速耗电的主导驱动因素。

- **敏感性与不确定性（SENSITIVITY & UNCERTAINTY）**：  
  分析当模型假设、参数取值与使用模式波动发生变化时，预测结果如何改变；量化不确定性并识别模型失效情形。

- **建议（RECOMMEND）**：  
  将模型洞见转化为对用户的节能建议与对操作系统电源管理策略的启示；讨论电池老化对结果的影响，以及模型框架对其他便携设备的可扩展性。

为支持参数估计与模型验证，我们采用 `MCM_2026_A/Database/` 作为主数据集：其中每个 CSV 以设备标识命名，记录秒级采样的电池与使用环境变量（如时间戳、电量百分比、温度、电压、电流、网络类型、亮屏状态、充电状态、前台应用等）；`Activation_Date_Phone.xlsx` 提供设备激活日期信息，可用于刻画电池使用年限/老化差异。

（第 1 章后续内容留白）

### 2 符号记述与问题假设

#### 2.1 符号与记号表（Notation）

| 记号 | 含义 | 单位/取值 | 数据来源/说明 |
| --- | --- | --- | --- |
| \(t\) | 连续时间 | s（或以时间戳换算） | 由 `timestamp_ms`（epoch 毫秒）换算 |
| \(SOC(t)\) | 电量状态（State of Charge） | \([0,1]\) | 由电量百分比近似：\(SOC \approx \\text{level}/100\) |
| \(C_{\\text{nom}}\) | 电池标称容量 | mAh | 日志列 `battery_capacity_mAh` |
| \(C_{\\text{eff}}\) | 电池有效容量（考虑温度/老化后的等效容量） | mAh | 待估计/分组拟合（与激活日期相关） |
| \(I(t)\) | 电池电流（放电为正的约定电流） | mA | 日志列 `battery_current_mA`（需结合充电状态解释符号/方向） |
| \(V(t)\) | 电池端电压 | mV | 日志列 `battery_voltage_mV` |
| \(T(t)\) | 电池温度 | ℃ | 日志列 `battery_temp_C` |
| \(u_{\\text{scr}}(t)\) | 亮屏指示变量 | \(\{0,1\}\) | 日志列 `screen_on`（True/False） |
| \(b(t)\) | 屏幕亮度（归一化） | \([0,1]\) | 建模变量；数据验证阶段可视为常数 \(b(t)\equiv b_0\) |
| \(u_{\\text{chg}}(t)\) | 充电指示变量 | \(\{0,1\}\) | 日志列 `is_charging`（True/False） |
| \(u_{\\text{net}}(t)\) | 网络模式 | {wi‑fi, mobile, none, …} | 日志列 `network_type` |
| \(a(t)\) | 前台应用（或活动标识） | 字符串 | 日志列 `foreground_app`（用于行为分段/负载代理） |
| \(\Delta t_k\) | 相邻采样时间间隔 | s | 由时间戳差分得到（采样近似 1–2 s） |
| \(\tau_{\\text{empty}}\) | 距离耗尽时间（time‑to‑empty） | s 或 h | 由模型预测得到 |
| \(I_{\\text{net}}(t)\) | 电池端净电流（净放电为正） | mA | 连续时间模型输入量（由负载与充电共同决定） |
| \(I_{\\text{d}}(t)\) | 纯放电电流（放电为正） | mA | 纯放电工况下的 \(I_{\\text{net}}(t)\)（满足 \(I_{\\text{d}}(t)\\ge 0\)） |
| \(\eta(t)\) | 库伦效率（充/放电效率） | \((0,1]\) | 可取常数或分段常数 |
| \(\Theta(t)\) | 绝对温度 | K | \(\Theta(t)=T(t)+273.15\) |
| \(T_{\min},T_{\max}\) | 舒适工作温度范围的下/上界 | ℃ | 待设定/待拟合（用于分段温度修正） |
| \(k_T(T)\) | 温度修正系数（乘性，影响放电速度） | \(\mathbb{R}_+\) | 工作温度内近似为 1，范围外指数惩罚 |

> 注：随着后续模型扩展（如引入温度修正项、老化参数、负载分解项等），本表将同步增补新符号以保持全文一致。

#### 2.2 问题假设（Assumptions）

题目要求建立连续时间、具有机理解释的电池耗电模型。由于题面并未提供固定数据库，下列假设仅针对“真实智能手机电池系统与使用情景”提出（而非对任何特定数据集字段作预设），以保证模型可定义、可求解、可用于预测与解释：

1. **SOC 与系统电量指示的映射假设**：在短时间尺度上，将系统显示的电量百分比视为 \(SOC(t)\) 的线性近似，即 \(SOC(t) \\approx \\text{battery\\_percent}(t)/100\)。  

2. **电池容量短期稳定假设**：在单次分析的短时间窗口内（如日内至数日），标称容量 \(C_{\\text{nom}}\) 可视为常数；温度与老化对容量的影响通过有效容量 \(C_{\\text{eff}}\) 的参数化体现，而不在该窗口内发生快速漂移。  

3. **电量守恒（库伦计）假设**：电池的 SOC 变化主要由净电流决定，满足  
   
   \[
   \dot{SOC}(t)= -\frac{I_{\text{net}}(t)}{C_{\text{eff}}}\cdot \eta(t),
   \]
   
   其中 \(I_{\text{net}}(t)>0\) 表示净放电、\(I_{\text{net}}(t)<0\) 表示净充电，\(\eta(t)\) 为充/放电库伦效率（可取常数或分段常数）。  

4. **分段近似假设（用于数值求解与验证）**：在短时间间隔 \(\Delta t_k\) 内，将外部负载与环境条件视为分段常值或分段线性，从而可对连续时间模型进行数值积分，并与离散观测进行对比。  

5. **负载可解释分解假设**：手机的总负载可视为若干可解释子负载的叠加（如屏幕、计算、网络、后台等），并允许用“模式/状态”来近似描述不同使用场景下的负载差异。  

6. **忽略自放电的假设**：在关注的时间尺度内（日内至数日），忽略自放电对 \(SOC(t)\) 的影响，相比设备负载功耗其量级可忽略。  


### 3 模型建立

#### 3.1 模型准备：最简放电方程

我们从对耗电的最简、可解释描述出发：在“纯放电”工况下（即设备未外接电源，电池只向负载供能），电量状态 \(SOC(t)\) 的变化应满足电量守恒。记电池有效容量为 \(C_{\\text{eff}}\)（单位 mAh），记净放电电流为 \(I_{\\text{d}}(t)\\ge 0\)（单位 mA），并令 \(\eta\\in(0,1]\) 表示放电库伦效率（若忽略效率损失则取 \(\eta=1\)）。

**引理 1（最简放电 SOC 方程）**  
在纯放电工况下，\(SOC(t)\) 满足如下常微分方程：

\[
\dot{SOC}(t)= -\frac{\eta\, I_{\\text{d}}(t)}{C_{\\text{eff}}}.
\]

**推论 1（time‑to‑empty 的基本表达式）**  
设初始时刻 \(t_0\) 的电量为 \(SOC(t_0)=SOC_0\in(0,1]\)。若在 \([t_0, t_0+\tau]\) 上持续纯放电，则 \(\tau_{\\text{empty}}\) 定义为使 \(SOC(t_0+\tau_{\\text{empty}})=0\) 的最小 \(\tau_{\\text{empty}}>0\)。由引理 1 得

\[
SOC_0=\int_{t_0}^{t_0+\tau_{\\text{empty}}}\frac{\eta\, I_{\\text{d}}(t)}{C_{\\text{eff}}}\,dt.
\]

特别地，若在该区间内可近似为常电流放电 \(I_{\\text{d}}(t)\equiv I_0\)，则

\[
\tau_{\\text{empty}}=\frac{SOC_0\,C_{\\text{eff}}}{\eta\, I_0}.
\]

以上给出了我们后续所有扩展模型（加入充电、温度、模式切换、老化等项）的“最小骨架”：任何复杂项都应以不违背该守恒结构为前提进行修正。

#### 3.2 “装饰项”一：亮屏、网络与温度修正（不考虑充电）

本节在不考虑充电过程的前提下，将“纯放电骨架”推广到可直接由数据验证的修正形式。记 \(u_{\text{scr}}(t)\in\{0,1\}\) 为亮屏指示变量（1=亮屏），记 \(u_{\text{net}}(t)\in\{\text{none},\text{wi-fi},\text{mobile}\}\) 为网络模式。记温度为 \(T(t)\)（℃），并取参考温度 \(T_{\text{ref}}\)（例如 30℃）。在本节中，我们将净放电电流 \(I_{\text{d}}(t)\) 以“放电强度”代理量表示为分段常值/分段线性的等效放电项，从而可在不显式使用充电信息的情况下拟合系数。

**修正系数的统一写法（乘性形式）**  
我们用三个无量纲修正系数表示亮屏、网络、温度对放电强度的相对影响：

\[
\dot{SOC}(t)= -\frac{\eta}{C_{\text{eff}}}\, I_0(t)\cdot k_{\text{scr}}\!\big(u_{\text{scr}}(t)\big)\cdot k_{\text{net}}\!\big(u_{\text{net}}(t)\big)\cdot k_T\!\big(T(t)\big),
\]

其中 \(I_0(t)\ge 0\) 为“基准放电强度”（可理解为在参考工况下的等效放电电流），三类装饰项分别定义如下。

**（1）亮屏—亮度修正：\(k_{\text{scr}}\)**  
在机理建模中，“息屏”与“亮屏”属于两种不同工作状态：亮屏会激活显示面板驱动、背光/发光与刷新链路等固定功耗，因此从息屏到亮屏应存在**不连续跳变**。在亮屏状态内部，亮度与显示功耗常用**线性（仿射）**近似建模。为同时体现“亮度连续、开关不连续”，我们定义亮度 \(b(t)\in[0,1]\)（0=最低亮度，1=最高亮度），并采用分段函数：

\[
k_{\text{scr}}\!\big(u_{\text{scr}},b\big)=
\begin{cases}
1, & u_{\text{scr}}=0\ (\text{息屏}),\\
1+\delta_{\text{scr}}+\beta_{\text{scr}}\,b, & u_{\text{scr}}=1\ (\text{亮屏}),
\end{cases}
\qquad (\delta_{\text{scr}}>0,\ \beta_{\text{scr}}\ge 0).
\]

其中 \(\delta_{\text{scr}}\) 刻画“息屏→亮屏”的固定跳变能耗，\(\beta_{\text{scr}}\) 刻画亮度的线性增量效应。该模型天然满足不连续性：即使取 \(b=0\)（最低亮度），仍有 \(k_{\text{scr}}(1,0)=1+\delta_{\text{scr}}>1\neq k_{\text{scr}}(0,\cdot)=1\)。

在数据验证阶段，若将亮度近似视为固定 \(b(t)\equiv b_0\)，则亮屏段系数退化为常数
\(\alpha_{\text{scr}}=1+\delta_{\text{scr}}+\beta_{\text{scr}}b_0\)，从而回到“息屏/亮屏两档系数”的简化形式。

**（2）网络修正：\(k_{\text{net}}\)**  
将网络模式分为三类（无网络/无线网络/蜂窝网络），用分段常数描述其相对能耗差异：

\[
k_{\text{net}}(u_{\text{net}})=
\begin{cases}
1, & u_{\text{net}}=\text{none}\\
\alpha_{\text{wifi}}, & u_{\text{net}}=\text{wi-fi}\\
\alpha_{\text{mob}}, & u_{\text{net}}=\text{mobile}
\end{cases}
\qquad (\alpha_{\text{wifi}}>0,\ \alpha_{\text{mob}}>0).
\]

**（3）温度修正：\(k_T\)**  
温度项我们采用“**存在舒适工作温度范围，范围外指数级惩罚**”的建模思路：在舒适范围内温度对放电速度的影响极弱（可近似为 1 或极小线性项）；当温度过低或过高时，电化学动力学与副反应对性能的影响显著增强，可用指数型关系描述。

为避免摄氏度带来的非线性歧义，先定义绝对温度 \(\Theta(t)=T(t)+273.15\)（单位 K），并取参考温度 \(\Theta_{\text{ref}}\)（对应 \(T_{\text{ref}}\)）。

**标准温度方程（Arrhenius 形式）**  
在电化学动力学中，许多“速率常数/扩散系数/界面反应速率”等量对温度的基本依赖可写为

\[
k(\Theta)=A\exp\!\left(-\frac{E_a}{R\Theta}\right),
\]

其中 \(E_a\) 为激活能，\(R\) 为气体常数。用参考温度消去常数 \(A\) 可得比值形式：

\[
\frac{k(\Theta)}{k(\Theta_{\text{ref}})}
=\exp\!\left(-\frac{E_a}{R}\left(\frac{1}{\Theta}-\frac{1}{\Theta_{\text{ref}}}\right)\right).
\]

**由 Arrhenius 推向“可建模温度修正”的推导**  
我们用两个最常见、且能直接落到放电速度上的通道来连接温度与 \(\dot{SOC}(t)\)：

1) **低温侧：内阻/极化随温度上升而下降（低温显著恶化）**  
在等效电路视角下，端电压可写为

\[
V(t)\approx OCV\!\big(SOC(t)\big)-I(t)\,R_{\text{int}}(\Theta)-V_{\text{pol}}(t),
\]

其中 \(R_{\text{int}}\)（含欧姆内阻与电荷转移阻抗等）随温度变化显著。若把界面反应“越快则阻抗越小”抽象为 \(R_{\text{int}}(\Theta)\propto 1/k(\Theta)\)，则

\[
\frac{R_{\text{int}}(\Theta)}{R_{\text{int}}(\Theta_{\text{ref}})}
=\exp\!\left(\frac{E_{a,1}}{R}\left(\frac{1}{\Theta}-\frac{1}{\Theta_{\text{ref}}}\right)\right),
\]

即温度越低（\(\Theta\) 越小），\(R_{\text{int}}\) 越大。

当以“到达截止电压 \(V_{\min}\)”定义放电终点时，恒流近似 \(I(t)\equiv I_0\) 下终止 SOC 满足

\[
OCV\!\big(SOC_{\text{end}}(\Theta)\big)\approx V_{\min}+I_0\,R_{\text{int}}(\Theta).
\]

在参考温度附近对 \(OCV(\cdot)\) 做一阶线性化（记 \(g=\frac{d\,OCV}{d\,SOC}>0\)）：

\[
SOC_{\text{end}}(\Theta)\approx SOC_{\text{end}}(\Theta_{\text{ref}})+\frac{I_0}{g}\Big(R_{\text{int}}(\Theta)-R_{\text{int}}(\Theta_{\text{ref}})\Big).
\]

于是“可用 SOC 窗口” \(\Delta SOC_{\text{use}}(\Theta)=SOC_0-SOC_{\text{end}}(\Theta)\) 随 \(R_{\text{int}}(\Theta)\) 增大而缩小。等价地，我们可将其吸收到“有效容量因子” \(g_T(\Theta)\in(0,1]\)：

\[
C_{\text{eff}}(\Theta)=C_{\text{eff}}\cdot g_T(\Theta),
\qquad
g_T(\Theta)=\frac{\Delta SOC_{\text{use}}(\Theta)}{\Delta SOC_{\text{use}}(\Theta_{\text{ref}})}.
\]

代回最简放电骨架

\[
\dot{SOC}(t)= -\frac{\eta\, I_0(t)}{C_{\text{eff}}(\Theta(t))},
\]

得到温度乘性修正系数

\[
k_T(\Theta)=\frac{C_{\text{eff}}}{C_{\text{eff}}(\Theta)}=\frac{1}{g_T(\Theta)}.
\]

当温度偏离不大时，\(g_T(\Theta)\) 可近似线性；当偏离显著、且 \(R_{\text{int}}(\Theta)\) 近似 Arrhenius 时，\(k_T(\Theta)\) 将呈指数型增长（尤其在低温侧）。

2) **高温侧：副反应/寄生电流随温度上升而上升（高温惩罚）**  
把高温导致的“额外不可用电量消耗”抽象为寄生电流 \(I_p(\Theta)\ge 0\)，常见的机理建模同样采用 Arrhenius：

\[
I_p(\Theta)=I_{p,\text{ref}}\exp\!\left(-\frac{E_{a,2}}{R}\left(\frac{1}{\Theta}-\frac{1}{\Theta_{\text{ref}}}\right)\right),
\]

其随温度上升而增大。于是净放电强度可写为 \(I_{\text{net}}=I_0+I_p(\Theta)\)，对应的放电速度乘子为

\[
k_{T,\text{high}}(\Theta)=\frac{I_0+I_p(\Theta)}{I_0}=1+\frac{I_p(\Theta)}{I_0},
\]

在高温侧给出“随温度上升加速”的惩罚项。

**最终可建模方程：工作温度窗 + 范围外指数惩罚（U 形）**  
将低温侧（内阻/可用容量）与高温侧（寄生消耗）合并，我们采用分段乘性温度系数（在舒适区取 1）：

\[
k_T(T)=
\begin{cases}
\exp\!\Big(\gamma_{\text{low}}\big(\frac{1}{\Theta}-\frac{1}{\Theta_{\min}}\big)\Big), & \Theta<\Theta_{\min},\\[6pt]
1+\beta_T\,(T-T_{\text{opt}}), & \Theta_{\min}\le \Theta\le \Theta_{\max},\\[6pt]
\exp\!\Big(\gamma_{\text{high}}\big(\frac{1}{\Theta_{\max}}-\frac{1}{\Theta}\big)\Big), & \Theta>\Theta_{\max},
\end{cases}
\qquad \Theta=T+273.15,
\]

其中 \(\gamma_{\text{low}},\gamma_{\text{high}}>0\)。若认为舒适区内影响可忽略，则取 \(\beta_T\approx 0\)，从而 \(k_T\) 在 \([\Theta_{\min},\Theta_{\max}]\) 内近似为 1，并在两侧随温度偏离呈指数上升；对应“放电效率”可定义为 \(\eta_T(T)=1/k_T(T)\)，其形状为以舒适区为顶的“U 形（或倒 U）窗口”。

在数据验证阶段，由于现有温度观测多落在常温附近，可将 \(\beta_T\) 视为极小并优先用数据估计 \(\gamma_{\text{low}},\gamma_{\text{high}}\) 是否显著；若温度范围不足以识别两侧指数，则可退化为舒适区内的弱线性近似。

**用于数据验证的等价回归形式**  
由于系统电量往往以百分比离散显示，直接对秒级 \(SOC(t)\) 求导会引入量化噪声。实践中可先对 \(SOC\) 做按分钟重采样，并用多分钟差分近似 \(\dot{SOC}(t)\)，再在放电段上拟合系数。例如，对 \(r(t)\equiv -\dot{SOC}(t)\)（单位 1/h）取对数可得到便于估计的线性回归：

\[
\log r(t)= c_0 + c_{\text{scr}}\,u_{\text{scr}}(t) + c_{\text{wifi}}\,\mathbf{1}\{u_{\text{net}}(t)=\text{wi-fi}\}+c_{\text{mob}}\,\mathbf{1}\{u_{\text{net}}(t)=\text{mobile}\} + c_T\,(T(t)-T_{\text{ref}}),
\]

其中 \(\mathbf{1}\{\cdot\}\) 为指示函数。该形式对应乘性修正系数的对数线性化，可直接用最小二乘或稳健回归估计参数，并用于检验“温度近似线性还是指数更合适”。