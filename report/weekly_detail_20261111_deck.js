// 本周汇报（详细版）：混合模型流水线建模、瓶颈诊断与优化（2026-11-11）
// Build: NODE_PATH=<dir with pptxgenjs> APPLY_THEME=<apply_theme.js> node build2.js out.pptx data.json
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const { applyTheme } = require(process.env.APPLY_THEME || "./apply_theme.js");

const OUT = process.argv[2] || "weekly_detail.pptx";
const D = JSON.parse(fs.readFileSync(process.argv[3] || "data.json"));

const HEX = { ink: "14213D", grey: "586174", light: "EDF1F5", white: "FFFFFF", blue: "2A78D6",
  amber: "E07A1F", green: "2E9E6B", red: "C8473D", mist: "9DB3D3", line: "DDE3EA" };
const THEME = { name: "Pipeline Weekly", headFontFace: "Microsoft YaHei", bodyFontFace: "Microsoft YaHei",
  colors: { dk1: HEX.ink, lt1: HEX.white, dk2: HEX.grey, lt2: HEX.light, accent1: HEX.blue, accent2: HEX.amber,
    accent3: HEX.green, accent4: HEX.red, accent5: HEX.mist, accent6: HEX.ink, hlink: HEX.blue, folHlink: HEX.grey } };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "混合模型流水线建模、瓶颈诊断与优化";
const C = pres.SchemeColor;

pres.defineSlideMaster({ title: "TITLE_DARK", background: { color: C.text1 }, objects: [
  { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.1, w: 11.7, h: 1.6, fontSize: 40, bold: true,
    color: C.background1, valign: "bottom", align: "left" }, text: "" } },
  { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.9, w: 11.7, h: 1.6, fontSize: 18,
    color: C.accent5, valign: "top", align: "left" }, text: "" } } ] });
pres.defineSlideMaster({ title: "CONTENT", background: { color: C.background1 }, margin: [0.5, 0.6, 0.5, 0.6],
  objects: [ { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.3, w: 12.1, h: 0.8, fontSize: 28,
    bold: true, color: C.text1, valign: "middle", align: "left", margin: 0 }, text: "" } } ],
  slideNumber: { x: 12.4, y: 7.05, w: 0.6, h: 0.3, fontSize: 10, color: C.text2, align: "right" } });

function content(title, section) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  return s;
}
function txt(s, text, o) {
  s.addText(text, Object.assign({ isTextBox: true, margin: 0, color: C.text1, fontSize: 14, valign: "top" }, o));
}
function card(s, x, y, w, h, fill) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: fill || C.background2 }, line: { type: "none" } });
}
function chip(s, x, y, label, fill, w) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: w || 1.1, h: 0.32, rectRadius: 0.16, fill: { color: fill }, line: { type: "none" } });
  txt(s, label, { x, y, w: w || 1.1, h: 0.32, fontSize: 11, bold: true, color: C.background1, align: "center", valign: "middle" });
}
function bullets(s, items, o) {
  txt(s, items.map((t, k) => (typeof t === "string"
    ? { text: t, options: { bullet: true, breakLine: k < items.length - 1 } }
    : { text: t[0], options: Object.assign({ bullet: true, breakLine: k < items.length - 1 }, t[1]) })),
    Object.assign({ paraSpaceAfter: 6 }, o));
}
function takeaway(s, text, y) {
  card(s, 0.6, y || 6.45, 12.1, 0.5, "E6EEF9");
  txt(s, text, { x: 0.85, y: (y || 6.45) + 0.08, w: 11.7, h: 0.36, fontSize: 14, bold: true, color: C.accent1, valign: "middle" });
}
const H = (t) => ({ text: t, options: { bold: true, color: HEX.white, fill: { color: HEX.ink }, fontSize: 12, align: "center", valign: "middle" } });
const T = (t, o) => ({ text: String(t), options: Object.assign({ fontSize: 12, align: "center", valign: "middle", color: HEX.ink }, o || {}) });
const G = { color: HEX.green, bold: true }, R = { color: HEX.red, bold: true }, B = { bold: true };
function table(s, rows, o) {
  s.addTable(rows, Object.assign({ border: { type: "solid", color: HEX.line, pt: 1 }, fill: { color: HEX.white }, margin: 0.04 }, o));
}
const chartBase = () => ({ catAxisLabelColor: HEX.grey, valAxisLabelColor: HEX.grey, catAxisLabelFontFace: "+mn-lt",
  valAxisLabelFontFace: "+mn-lt", catAxisLabelFontSize: 11, valAxisLabelFontSize: 10, dataLabelFontFace: "+mn-lt",
  dataLabelFontSize: 10, dataLabelColor: HEX.ink, legendFontFace: "+mn-lt", legendFontSize: 11, legendColor: HEX.grey,
  titleFontFace: "+mn-lt", titleFontSize: 12, titleColor: HEX.grey, valGridLine: { color: HEX.line, size: 0.75 },
  catGridLine: { style: "none" }, valAxisLineShow: false });

// =============================================================== 1
pres.addSection({ title: "开场" });
{
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "开场" });
  s.addText("混合模型流水线：建模、瓶颈诊断与优化", { placeholder: "title" });
  s.addText("本周进展  ·  Nemotron-H 9B（Mamba + attention 混合）  ·  双机 4 卡流水线（PP=4）  ·  vLLM 0.10.1.1\n2026-11-11", { placeholder: "body" });
  s.addNotes("见汇报稿第 1 页。");
}

// =============================================================== 2 key findings
{
  const s = content("本周结论：四个发现", "开场");
  const cards = [
    ["≤ 5%", "单段耗时预测误差", "按层类型数量建模；对没见过的切分也成立（留出切分 0.7–4.9%）", C.accent1],
    [`+${D.capture.thr8k}%`, "长输入：调大 graph 长度上限", `vLLM 超过 8192 token 关掉 CUDA graph；改一个参数，8K 吞吐 +${D.capture.thr8k}%、16K +${D.capture.thr16k}%，输出不变`, C.accent3],
    [`+${D.spmd.thr1k}~${D.spmd.thr8k}%`, "主进程瓶颈：SPMD + 修复 vLLM bug", "GPU 利用率约 25%，瓶颈是主进程单核 CPU；修好 SPMD 路径的 bug 后吞吐大幅提升，输出不变", C.accent3],
    ["10 s → 0.4 s", "换切分的 graph 重录", "重录时间 95% 是 Python 垃圾回收；虚拟内存可让 KV 地址不变、数据不丢", C.accent4],
  ];
  cards.forEach(([big, h, b, col], i) => {
    const x = 0.6 + (i % 2) * 6.1, y = 1.3 + Math.floor(i / 2) * 2.45;
    card(s, x, y, 5.95, 2.25);
    txt(s, big, { x: x + 0.3, y: y + 0.2, w: 5.4, h: 0.8, fontSize: 36, bold: true, color: col });
    txt(s, h, { x: x + 0.3, y: y + 1.0, w: 5.4, h: 0.4, fontSize: 16, bold: true });
    txt(s, b, { x: x + 0.3, y: y + 1.45, w: 5.4, h: 0.75, fontSize: 12, color: C.text2 });
  });
  takeaway(s, "建议：先采用两项无代价优化（graph 上限 + SPMD 修复）；在线换切分暂缓——开 graph 后默认均分已是最优。", 6.3);
  s.addNotes("见汇报稿第 2 页。");
}

// =============================================================== 3 setup
{
  const s = content("实验环境与方法", "开场");
  table(s, [
    [H("项目"), H("配置")],
    [T("硬件", B), T("2 台机器 × 2 张 RTX 3090（24 GB）；机器间万兆以太网（无 InfiniBand）", { align: "left" })],
    [T("软件", B), T("vLLM 0.10.1.1，V0 引擎，Ray 跨机；插件修复 NemotronH 流水线 4 个 bug（+ 本周新修 1 个）", { align: "left" })],
    [T("模型", B), T("Nemotron-Nano-9B-v2：56 层 = 27 Mamba + 25 MLP + 4 attention（第 14/21/30/39 层）", { align: "left" })],
    [T("流水线", B), T("PP=4：段 1、2 在 slave21，段 3、4 在 slave20；段 2→3 跨机", { align: "left" })],
    [T("负载", B), T("输入 1K / 8K / 16K，输出 128，并发 16，32 条请求；max_num_seqs 16", { align: "left" })],
  ], { x: 0.6, y: 1.3, w: 7.3, colW: [1.2, 6.1], rowH: 0.62 });
  card(s, 8.2, 1.3, 4.5, 4.95);
  txt(s, "测量与验证原则", { x: 8.45, y: 1.5, w: 4.0, h: 0.4, fontSize: 16, bold: true });
  bullets(s, [
    "GPU 时间用 CUDA event，CPU 时间单独记录，不混算",
    "建模前预留验证切分 15,13,15,13，不参与拟合",
    "关键对比重复 3–4 次，Welch t 检验",
    "每项优化都做输出正确性检查：greedy 400 token，含 9.7K–15K token 长文检索题",
    "跨机 trace 用 rsync 回收后统一分析",
  ], { x: 8.45, y: 2.0, w: 4.0, h: 4.1, fontSize: 13 });
  takeaway(s, "今天共约 80 次四卡实测；所有结论都附重复次数与正确性检查。", 6.45);
  s.addNotes("见汇报稿第 3 页。");
}

// =============================================================== 4 overview
{
  const s = content("本周计划七步：完成情况", "开场");
  table(s, [
    [H("步骤"), H("状态"), H("主要结果"), H("还差什么")],
    [T("1 测量口径", B), T("部分", { color: HEX.amber, bold: true }), T("GPU/CPU 分开计时；发现 graph 回退、主进程瓶颈", { align: "left" }), T("micro-batch 唯一编号", { align: "left" })],
    [T("2 计算模型", B), T("初版", { color: HEX.blue, bold: true }), T("按层类型数量回归；留出误差 0.7–4.9%", { align: "left" }), T("单层级预测", { align: "left" })],
    [T("3 显存模型", B), T("完成", G), T("权重误差 ≤0.01 GiB；可变性源码核查", { align: "left" }), T("过渡峰值实测", { align: "left" })],
    [T("4 通信模型", B), T("完成", G), T("机内 18.6 GB/s，跨机 1.18 GB/s", { align: "left" }), T("用真实 trace 校准", { align: "left" })],
    [T("5 流水线模拟", B), T("初版", { color: HEX.blue, bold: true }), T("V0 调度事件模拟；加入随上下文增长的 CPU 开销", { align: "left" }), T("TTFT 偏低；混合步骤", { align: "left" })],
    [T("6 切分搜索", B), T("完成", G), T("6545 种切分 → 容量筛选 → 模拟 → 实测", { align: "left" }), T("—", { align: "left" })],
    [T("7 验证 + 重配置成本", B), T("完成", G), T("切分收益已量化；切换成本拆解并给出降本方案", { align: "left" }), T("端到端在线切换未实现", { align: "left" })],
  ], { x: 0.6, y: 1.3, w: 12.1, colW: [2.1, 0.9, 5.6, 3.5], rowH: 0.6 });
  takeaway(s, "计划外收获：诊断出两个真实瓶颈并给出已验证的优化（graph 上限、SPMD 修复）。", 6.3);
  s.addNotes("见汇报稿第 4 页。");
}

// =============================================================== 5 compute model
pres.addSection({ title: "建模" });
{
  const s = content("计算模型：按层类型数量预测每段耗时", "建模");
  card(s, 0.6, 1.3, 6.0, 4.9);
  txt(s, "做法", { x: 0.85, y: 1.45, w: 5.5, h: 0.4, fontSize: 16, bold: true });
  bullets(s, [
    "每段耗时 = Σ 各类层数量 ×（固定成本 + 随工作量增长的成本）",
    "decode：工作量 = batch 大小；attention 另加 × 历史长度之和",
    "prefill：工作量 = 本批 token 数；attention 另加 × 长度平方和",
    "decode 的 graph / eager 两种模式分开拟合（相差 2–4 倍）",
    "所有段、所有切分共享一组系数 → 能给没跑过的切分定价",
  ], { x: 0.85, y: 1.95, w: 5.5, h: 2.8, fontSize: 13 });
  txt(s, "目标值用 GPU 时间：graph 模式下 CPU 执行时间不可用（非末段 launch 后约 1 ms 就返回）", { x: 0.85, y: 4.85, w: 5.5, h: 0.7, fontSize: 12, color: C.text2 });
  txt(s, "未做到：单个层级别的预测（目前只到整段）", { x: 0.85, y: 5.6, w: 5.5, h: 0.4, fontSize: 12, bold: true, color: C.accent4 });
  table(s, [
    [H("误差（中位数）"), H("默认配置·训练"), H("默认配置·留出"), H("开 graph·留出")],
    [T("decode（graph）", B), T("1.3%"), T("1.1%", G), T("1.8%", G)],
    [T("decode（eager）", B), T("4.4%"), T("4.9%", G), T("—")],
    [T("prefill", B), T("0.7%"), T("0.7%", G), T("0.8%", G)],
  ], { x: 6.9, y: 1.3, w: 5.8, colW: [1.9, 1.3, 1.3, 1.3], rowH: 0.55 });
  s.addChart(pres.charts.BAR, [{ name: "误差", labels: ["decode graph", "decode eager", "prefill"], values: [1.1, 4.9, 0.7] }],
    Object.assign(chartBase(), { x: 6.9, y: 3.6, w: 5.8, h: 2.6, barDir: "col", chartColors: [HEX.blue], showTitle: true,
      title: "留出切分 15,13,15,13 上的每段误差（%）", showValue: true, dataLabelPosition: "outEnd",
      dataLabelFormatCode: '0.0"%"', valAxisMinVal: 0, valAxisMaxVal: 6, valAxisMajorUnit: 2, showLegend: false, barGapWidthPct: 90 }));
  takeaway(s, "单段计算模型是本周最扎实的部分：对新切分误差 ≤5%。", 6.45);
  s.addNotes("见汇报稿第 5 页。");
}

// =============================================================== 6 memory
{
  const s = content("显存模型与可变性：权重估得准，KV 与 Mamba 状态启动后固定", "建模");
  const st = ["均分 段1", "段2", "段3", "段4", "18,10,10,18 段1", "段2", "段3", "段4"];
  s.addChart(pres.charts.BAR, [
    { name: "权重", labels: st, values: [5.96, 5.62, 5.62, 5.95, 6.86, 4.72, 4.70, 6.86] },
    { name: "运行时临时空间", labels: st, values: [4.08, 4.06, 4.06, 4.14, 4.16, 3.84, 3.90, 4.16] },
    { name: "留给 KV", labels: st, values: [11.15, 11.52, 11.52, 11.11, 10.17, 12.64, 12.59, 10.17] },
  ], Object.assign(chartBase(), { x: 0.6, y: 1.25, w: 5.9, h: 5.0, barDir: "col", barGrouping: "stacked",
    chartColors: [HEX.ink, HEX.mist, HEX.blue], showLegend: true, legendPos: "b", showTitle: true, title: "每张卡显存去向（GiB）",
    showValue: true, dataLabelPosition: "ctr", dataLabelColor: HEX.white, dataLabelFormatCode: "0.0", valAxisMaxVal: 24,
    valAxisMajorUnit: 4, catAxisLabelFontSize: 9, barGapWidthPct: 40 }));
  table(s, [
    [H("显存"), H("何时分配"), H("能否运行中改")],
    [T("权重", B), T("启动加载", { align: "left" }), T("sleep 可整块卸载/恢复；不能加新层", { align: "left" })],
    [T("attention KV", B), T("启动时一次性；每 VE 每层一个 torch.zeros", { align: "left" }), T("不能缩放；释放 block 不还显存", R)],
    [T("Mamba 状态", B), T("首次 forward；按 max_num_seqs", { align: "left" }), T("不能缩放；sleep 也释放不了", R)],
    [T("临时空间", B), T("按最大 prefill 批次预留 ~4.1 GiB", { align: "left" }), T("随 max_model_len 变", { align: "left" })],
  ], { x: 6.8, y: 1.3, w: 5.9, colW: [1.25, 2.35, 2.3], rowH: 0.75 });
  bullets(s, ["权重估计 vs 实测：误差 ≤ 0.01 GiB（4 段 × 2 切分）", "V0 的 KV 池按各段最小空余定大小 → 两端加重会挤小整体容量", "当前负载下 6545 种切分全部装得下"],
    { x: 6.8, y: 5.15, w: 5.9, h: 1.2, fontSize: 12 });
  s.addNotes("见汇报稿第 6 页。");
}

// =============================================================== 7 comm
{
  const s = content("通信模型：跨机器比机器内慢 16 倍", "建模");
  const lb = ["32 token（decode）", "512", "2048", "8192（prefill）"];
  s.addChart(pres.charts.BAR, [
    { name: "同一台机器内", labels: lb, values: [0.09, 0.57, 2.1, 8.2] },
    { name: "跨两台机器", labels: lb, values: [0.63, 7.9, 31.3, 124.9] },
  ], Object.assign(chartBase(), { x: 0.6, y: 1.25, w: 7.0, h: 5.0, barDir: "col", chartColors: [HEX.mist, HEX.amber],
    showLegend: true, legendPos: "b", showTitle: true, title: "传一次段间结果（hidden + residual，bf16）的耗时（ms）",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", valAxisMaxVal: 140, valAxisMajorUnit: 20, barGapWidthPct: 60 }));
  table(s, [
    [H("拟合 t = a + 字节/带宽"), H("固定延迟 a"), H("带宽")],
    [T("机器内（GPU0↔GPU1）", B), T("0.11 ms"), T("18.6 GB/s")],
    [T("跨机（万兆网，NCCL socket）", B), T("0.12 ms"), T("1.18 GB/s", R)],
  ], { x: 7.9, y: 1.3, w: 4.8, colW: [2.2, 1.3, 1.3], rowH: 0.6 });
  bullets(s, [
    "跨机已达线速 94%，网络本身是上限",
    "decode 每步跨机 < 1 ms，可忽略",
    "prefill 2048 token 跨机 31 ms，与一段计算同量级 → 切点位置影响长输入",
    "单向计时与往返减半一致：孤立测量没有混入等待",
  ], { x: 7.9, y: 3.35, w: 4.8, h: 2.8, fontSize: 13 });
  s.addNotes("见汇报稿第 7 页。");
}

// =============================================================== 8 simulator
{
  const s = content("流水线模拟器：吞吐与每字间隔可用，首字时间偏低", "建模");
  card(s, 0.6, 1.3, 5.2, 2.7);
  txt(s, "模拟了 V0 的什么", { x: 0.85, y: 1.45, w: 4.7, h: 0.4, fontSize: 15, bold: true });
  bullets(s, ["4 个虚拟引擎（VE）各自调度，新请求分给最空闲的 VE", "每个 VE 同时只有一个批次在流水线中", "每段一次只算一批；段间按通信模型传数据", "每段 CPU 开销 = 4.0 ms + 0.45 ms × 上下文千 token 数"],
    { x: 0.85, y: 1.95, w: 4.7, h: 2.0, fontSize: 12 });
  card(s, 0.6, 4.2, 5.2, 2.05);
  txt(s, "待解决", { x: 0.85, y: 4.35, w: 4.7, h: 0.4, fontSize: 15, bold: true, color: C.accent4 });
  bullets(s, ["TTFT 偏低 16–26%：prefill 有未建模开销，需加 micro-batch 编号拆分等待与传输", "chunked prefill 的混合步骤尚未建模"],
    { x: 0.85, y: 4.8, w: 4.7, h: 1.4, fontSize: 12 });
  table(s, [
    [H("留出切分 15,13,15,13（开 graph）"), H("吞吐"), H("TPOT"), H("TTFT")],
    [T("1K：固定开销模型", B), T("+4.8%"), T("−2.4%"), T("−16%", R)],
    [T("8K：固定开销模型", B), T("+51%", R), T("−43%", R), T("−19%", R)],
    [T("1K：开销随上下文增长", B), T("+6.3%", G), T("−4.5%", G), T("−16%", R)],
    [T("8K：开销随上下文增长", B), T("+10%", G), T("−5.3%", G), T("−19%", R)],
  ], { x: 6.1, y: 1.3, w: 6.6, colW: [3.0, 1.2, 1.2, 1.2], rowH: 0.6 });
  txt(s, "数字 = 预测相对实测的误差。固定开销模型在 8K 上失效，揭示出“主进程 CPU 开销随上下文长度增长”（见第 13 页）。",
    { x: 6.1, y: 4.45, w: 6.6, h: 0.8, fontSize: 12, color: C.text2 });
  takeaway(s, "模型的失败本身也是证据：它指出了主进程这一瓶颈。", 6.45);
  s.addNotes("见汇报稿第 8 页。");
}

// =============================================================== 9 partitions default config
pres.addSection({ title: "切分验证" });
{
  const s = content("切分验证（vLLM 默认配置）：推荐切分只在长输入占优", "切分验证");
  const lb = ["短输入 1K", "长输入 8K", "超长 16K"];
  s.addChart(pres.charts.BAR, [
    { name: "默认均分 14,14,14,14", labels: lb, values: [60.8, 195.6, 305.2] },
    { name: "模型推荐 7,17,18,14", labels: lb, values: [65.2, 181.1, 286.3] },
  ], Object.assign(chartBase(), { x: 0.6, y: 1.25, w: 5.8, h: 5.0, barDir: "col", chartColors: [HEX.mist, HEX.amber],
    showLegend: true, legendPos: "b", showTitle: true, title: "每字间隔 TPOT（ms，越低越好）", showValue: true,
    dataLabelPosition: "outEnd", dataLabelFormatCode: "0", valAxisMaxVal: 350, valAxisMajorUnit: 50, barGapWidthPct: 60 }));
  table(s, [
    [H("负载"), H("指标"), H("默认 均值±sd"), H("推荐 均值±sd"), H("差异"), H("t")],
    [T("1K", B), T("吞吐"), T("220.5±1.3"), T("207.6±0.1"), T("−5.8%", R), T("−20.0")],
    [T("1K", B), T("TPOT"), T("60.8±0.5"), T("65.2±0.1"), T("+7.3%", R), T("17.2")],
    [T("8K", B), T("吞吐"), T("58.2±2.0"), T("61.2±1.6"), T("+5.0%"), T("2.2")],
    [T("8K", B), T("TPOT"), T("195.6±4.5"), T("181.1±2.2"), T("−7.5%", G), T("−5.6")],
    [T("16K", B), T("吞吐"), T("43.1"), T("44.4"), T("+3%"), T("n=1")],
  ], { x: 6.7, y: 1.3, w: 6.0, colW: [0.7, 0.8, 1.3, 1.3, 0.95, 0.95], rowH: 0.55 });
  txt(s, "默认 n=4，推荐 n=3（8K/1K）；16K 各 1 次。推荐切分来自 6545 种切分的模型搜索。", { x: 6.7, y: 4.75, w: 6.0, h: 0.6, fontSize: 12, color: C.text2 });
  takeaway(s, "最优切分似乎随负载变化，收益 5–7%——但下一页说明，这个收益来自 graph 被关掉。", 6.45);
  s.addNotes("见汇报稿第 9 页。");
}

// =============================================================== 10 why rank0 slow
{
  const s = content("原因：超过 8192 token 关掉 CUDA graph，第 1 张卡最吃亏", "切分验证");
  s.addChart(pres.charts.BAR, [{ name: "decode GPU 忙碌", labels: ["段 1", "段 2", "段 3", "段 4"], values: [41.0, 14.6, 14.8, 16.0] }],
    Object.assign(chartBase(), { x: 0.6, y: 1.25, w: 5.0, h: 5.0, barDir: "col", chartColors: [HEX.amber, HEX.mist, HEX.mist, HEX.mist],
      showLegend: false, showTitle: true, title: "默认均分（每段 14 层），16K，eager（秒）", showValue: true,
      dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", valAxisMaxVal: 50, valAxisMajorUnit: 10, barGapWidthPct: 70 }));
  const pts = [
    ["vLLM 的规则", "录 graph 时 KV 地址表按 max_seq_len_to_capture = 8192 定长分配；任一序列超过 8192，整步退回 eager（model_runner.py:743–751, 1008–1019）"],
    ["混合模型特别吃亏", "56 层里只有 4 层 attention 需要地址表，但 graph 是整段录的：4 层超限，另外 52 层也跟着 eager，每段慢 2–4 倍"],
    ["为什么段 1 最慢", "段 1 与 vLLM 引擎在同一进程抢 CPU；eager 每个 kernel 都要 CPU 下命令。同样 14 层，段 1 慢 2.8 倍"],
    ["所以推荐切分有效", "“段 1 少放层”是在帮它躲 eager 的 CPU 开销，不是层分布本身更好"],
  ];
  pts.forEach(([h, b], i) => {
    const y = 1.3 + i * 1.25;
    s.addShape(pres.shapes.OVAL, { x: 5.95, y: y + 0.05, w: 0.45, h: 0.45, fill: { color: C.text1 }, line: { type: "none" } });
    txt(s, String(i + 1), { x: 5.95, y: y + 0.05, w: 0.45, h: 0.45, fontSize: 14, bold: true, color: C.background1, align: "center", valign: "middle" });
    txt(s, h, { x: 6.6, y, w: 6.1, h: 0.4, fontSize: 15, bold: true });
    txt(s, b, { x: 6.6, y: y + 0.42, w: 6.1, h: 0.8, fontSize: 12, color: C.text2 });
  });
  s.addNotes("见汇报稿第 10 页。");
}

// =============================================================== 11 capture fix
pres.addSection({ title: "优化" });
{
  const s = content("优化 1：调大 graph 长度上限，长输入吞吐 +13~21%", "优化");
  table(s, [
    [H("负载"), H("指标"), H("上限 8192"), H("上限 = 最大上下文"), H("变化"), H("t")],
    [T("8K", B), T("吞吐 tok/s"), T("58.2±2.0（n=4）"), T("70.7±3.0（n=3）"), T("+21.4%", G), T("6.3")],
    [T("8K", B), T("TPOT ms"), T("195.6±4.5"), T("147.9±5.3"), T("−24.4%", G), T("−12.6")],
    [T("16K", B), T("吞吐 tok/s"), T("42.5±0.6（n=3）"), T("48.0±0.4（n=3）"), T("+12.8%", G), T("12.1")],
    [T("16K", B), T("TPOT ms"), T("310.7±5.7"), T("272.1±2.9"), T("−12.4%", G), T("−10.4")],
    [T("两者", B), T("TTFT"), T("—"), T("—"), T("不变"), T("≈0")],
  ], { x: 0.6, y: 1.3, w: 7.2, colW: [0.6, 1.05, 1.85, 1.85, 1.0, 0.85], rowH: 0.55 });
  card(s, 8.1, 1.3, 4.6, 4.95);
  txt(s, "做法与验证", { x: 8.35, y: 1.45, w: 4.1, h: 0.4, fontSize: 15, bold: true });
  bullets(s, [
    "启动参数 --max-seq-len-to-capture = max_model_len，不改代码",
    "长输入 decode 从全部 eager 变为全部 graph",
    "graph 录制时间、显存、KV 预算都不变",
    "段 1 不再慢（7.4 s vs 其他 6.5–8.1 s）",
    "正确性：7 条提示（含 4 条 9.7K–15K 检索题）6/7 逐字一致；1 条思考文字差一个词、答案相同",
    "16K 提升较小：chunked prefill 的混合步骤仍不走 graph",
  ], { x: 8.35, y: 1.95, w: 4.1, h: 4.2, fontSize: 12 });
  txt(s, "TTFT 不变符合预期：prefill 本来就不走 graph。", { x: 0.6, y: 4.75, w: 7.2, h: 0.4, fontSize: 12, color: C.text2 });
  takeaway(s, "零代价优化：一个启动参数，长输入吞吐 +13~21%，输出不变。", 6.45);
  s.addNotes("见汇报稿第 11 页。");
}

// =============================================================== 12 partitions with graph
{
  const s = content("开 graph 后重比 7 种切分：默认均分就是最优", "优化");
  const P = ["默认 14×4", "15,13,15,13*", "20,12,12,12", "16,12,12,16", "12,16,16,12", "18,10,10,18", "12,12,12,20"];
  s.addChart(pres.charts.BAR, [
    { name: "1K", labels: P, values: [0, -0.3, -1.3, -1.9, -3.0, -5.7, -6.8] },
    { name: "8K", labels: P, values: [0, -0.5, -0.8, -1.9, 1.6, -1.5, -4.1] },
  ], Object.assign(chartBase(), { x: 0.6, y: 1.25, w: 7.4, h: 5.0, barDir: "col", chartColors: [HEX.mist, HEX.blue],
    showLegend: true, legendPos: "b", showTitle: true, title: "吞吐相对默认均分（%）", showValue: true,
    dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", valAxisMinVal: -8, valAxisMaxVal: 3, valAxisMajorUnit: 2,
    catAxisLabelFontSize: 10, barGapWidthPct: 50 }));
  card(s, 8.3, 1.3, 4.4, 4.95);
  txt(s, "结论", { x: 8.55, y: 1.45, w: 3.9, h: 0.4, fontSize: 15, bold: true });
  bullets(s, [
    "默认均分在 1K 最优；8K 上 12,16,16,12 高 1.6%，在噪声（约 3%）内",
    "两端加重的切分最差（−4 ~ −7%）",
    "之前推荐的 7,17,18,14 开 graph 后反而差 9–11%",
    "单段模型仍准确：留出 decode 1.8%、prefill 0.8%",
    "* 15,13,15,13 为预留验证切分",
  ], { x: 8.55, y: 1.95, w: 3.9, h: 4.2, fontSize: 12 });
  takeaway(s, "开 graph 后切分没有可测收益——真正的瓶颈不在 GPU 上的层怎么分。", 6.45);
  s.addNotes("见汇报稿第 12 页。");
}

// =============================================================== 13 driver bottleneck
{
  const s = content("发现 2：瓶颈是 vLLM 主进程的单核 CPU", "优化");
  table(s, [
    [H("段（8K，开 graph）"), H("GPU 计算"), H("一次调用"), H("其中 send"), H("调用间空闲")],
    [T("段 1（与引擎同进程）", B), T("6.1 ms"), T("25.4 ms", R), T("20.8 ms", R), T("0.3 ms", R)],
    [T("段 2", B), T("5.9 ms"), T("10.3 ms"), T("3.9 ms"), T("15.5 ms")],
    [T("段 3（跨机）", B), T("5.6 ms"), T("6.4 ms"), T("1.0 ms"), T("19.7 ms")],
    [T("段 4", B), T("7.1 ms"), T("14.0 ms"), T("—"), T("11.9 ms")],
  ], { x: 0.6, y: 1.3, w: 7.0, colW: [2.2, 1.15, 1.25, 1.2, 1.2], rowH: 0.55 });
  table(s, [
    [H("同 batch=4"), H("GPU"), H("段 1 一次调用"), H("段 1 send 等待")],
    [T("1K 上下文", B), T("6.0 ms"), T("14.0 ms"), T("10.0 ms")],
    [T("8K 上下文", B), T("6.1 ms"), T("25.3 ms", R), T("20.3 ms", R)],
  ], { x: 0.6, y: 4.3, w: 7.0, colW: [2.2, 1.4, 1.7, 1.7], rowH: 0.5 });
  card(s, 7.9, 1.3, 4.8, 4.95);
  txt(s, "证据链", { x: 8.15, y: 1.45, w: 4.3, h: 0.4, fontSize: 15, bold: true });
  bullets(s, [
    "段 1 一刻不停，其他段各空等 12–20 ms",
    "对齐段 1/段 2 时间戳：段 2 在段 1 开始 send 后 13.7 ms 才被调起，真正接收只要 2.3 ms",
    "运行中 top -H：主进程主线程 CPU 94–100%",
    "上下文越长，主进程每步要打包的元数据越多，开销越大",
    "估算 GPU 利用率约 25%",
  ], { x: 8.15, y: 1.95, w: 4.3, h: 4.2, fontSize: 12 });
  takeaway(s, "调度、段 1 计算、给其他段派活都挤在一个 Python 进程里；派活慢 → 段 1 卡在 send → 更慢。", 6.45);
  s.addNotes("见汇报稿第 13 页。");
}

// =============================================================== 14 SPMD
{
  const s = content(`优化 2：SPMD 模式 + 修复 vLLM bug，吞吐 +${D.spmd.thr1k}~${D.spmd.thr8k}%`, "优化");
  const lb = ["1K 吞吐", "8K 吞吐"];
  s.addChart(pres.charts.BAR, [
    { name: "常规", labels: lb, values: [D.spmd.base1k, D.spmd.base8k] },
    { name: "SPMD + 修复", labels: lb, values: [D.spmd.spmd1k, D.spmd.spmd8k] },
  ], Object.assign(chartBase(), { x: 0.6, y: 1.25, w: 4.6, h: 5.0, barDir: "col", chartColors: [HEX.mist, HEX.green],
    showLegend: true, legendPos: "b", showTitle: true, title: `吞吐（tok/s，各 ${D.spmd.n} 次均值）`, showValue: true,
    dataLabelPosition: "outEnd", dataLabelFormatCode: "0", barGapWidthPct: 60 }));
  card(s, 5.5, 1.3, 7.2, 2.75);
  txt(s, "vLLM V0 的 bug（SPMD 路径漏传两个参数）", { x: 5.75, y: 1.45, w: 6.8, h: 0.4, fontSize: 15, bold: true, color: C.accent4 });
  bullets(s, [
    "finished_requests_ids 缺失 → Mamba 状态无法回收，第一步就崩溃",
    "virtual_engine 缺失 → 所有 VE 回放 VE0 的 graph，读写错误的 KV（静默算错）",
    "修复：插件 spmd_fix.py，按常规路径补齐两个参数",
  ], { x: 5.75, y: 1.95, w: 6.8, h: 2.0, fontSize: 12 });
  table(s, [
    [H("指标"), H("常规"), H("SPMD + 修复"), H("变化")],
    [T("1K TPOT", B), T(`${D.spmd.tpot_base1k} ms`), T(`${D.spmd.tpot_spmd1k} ms`), T(`${D.spmd.tpot1k}%`, G)],
    [T("8K TPOT", B), T(`${D.spmd.tpot_base8k} ms`), T(`${D.spmd.tpot_spmd8k} ms`), T(`${D.spmd.tpot8k}%`, G)],
    [T("1K TTFT", B), T("1435 ms"), T("1628 ms"), T("+13%", R)],
    [T("正确性", B), T("—"), T("串行 7/7、并发 7/7 逐字一致", G), T("—")],
  ], { x: 5.5, y: 4.1, w: 7.2, colW: [1.4, 1.5, 3.1, 1.2], rowH: 0.42 });
  takeaway(s, "各 3 次，t>10；8K TTFT −4%，但 1K TTFT 变慢 13%（原因待查）。只需两个环境变量 + 插件修复。", 6.45);
  s.addNotes("见汇报稿第 14 页。");
}

// =============================================================== 15 chunk
{
  const s = content("chunked prefill：默认 2048 已最优，混合步骤靠调参解决不了", "优化");
  table(s, [
    [H("16K 输入，开 graph"), H("吞吐"), H("TTFT"), H("TPOT"), H("临时空间 / KV（GiB/卡）")],
    [T("chunk 2048（默认，n=3）", B), T("48.0", G), T("7.3 s", G), T("272 ms"), T("~1.5 / ~13.8")],
    [T("chunk 4096", B), T("48.4（+1%）"), T("7.7 s"), T("268 ms"), T("1.5–1.7 / 13.4–13.9")],
    [T("chunk 8192", B), T("43.9（−9%）", R), T("11.0 s", R), T("289 ms"), T("2.3–2.4 / 12.7–13.1")],
    [T("关闭 chunked prefill", B), T("40.2（−16%）", R), T("19.9 s", R), T("244 ms", G), T("8.4–8.6 / 6.5–6.9", R)],
  ], { x: 0.6, y: 1.3, w: 7.6, colW: [2.05, 1.6, 0.95, 1.0, 2.0], rowH: 0.6 });
  card(s, 8.5, 1.3, 4.2, 4.95);
  txt(s, "怎么理解", { x: 8.75, y: 1.45, w: 3.7, h: 0.4, fontSize: 15, bold: true });
  bullets(s, [
    "chunked prefill：长输入切块，每步“读一块 + 推进生成”混合执行",
    "好处：生成不被长时间阻塞、TTFT 短、临时空间小",
    "代价：混合步骤形状不固定，不能用 graph",
    "关掉：生成全走 graph（TPOT −10%），但 TTFT ×2.7、KV 空间减半",
    "要消除代价，需让混合步骤也能用 graph（改执行方式，如分段 graph）",
  ], { x: 8.75, y: 1.95, w: 3.7, h: 4.2, fontSize: 12 });
  takeaway(s, "保持默认；混合步骤的 graph 化是后续需要改 vLLM 执行方式的工作（预估上限约 10%）。", 6.45);
  s.addNotes("见汇报稿第 15 页。");
}

// =============================================================== 16 teacher Q: cost breakdown
pres.addSection({ title: "回答老师" });
{
  const s = content("老师的问题：显存用途动态切换的中断，到底贵在哪", "回答老师");
  txt(s, "问题：同一块显存，这一刻放 KV，下一刻放模型参数；vLLM 预先占一大块 PyTorch 张量做 KV，释放再申请会不会造成中断？vLLM 考虑过吗？",
    { x: 0.6, y: 1.2, w: 12.1, h: 0.6, fontSize: 13, color: C.text2 });
  table(s, [
    [H("切换的环节"), H("原来（vLLM V0 做法）"), H("是否主因"), H("现在能做到")],
    [T("释放 KV 张量 / 申请新显存", B), T("4–9 ms"), T("否"), T("VMM 部分收缩约 54 ms，但地址不变")],
    [T("KV 内容", B), T("全部丢失，在途请求要重算或先迁移", R), T("是"), T("VMM 只让出部分页，其余 KV 完好", G)],
    [T("CUDA graph", B), T("KV 地址变 → 全部失效，重录 7–14 s", R), T("最大头"), T("跳过 GC 重录 0.4 s；VMM 保地址后仅层变化的段需重录", G)],
    [T("搬新层权重（3 层，pinned）", B), T("14–47 ms"), T("中"), T("同左")],
    [T("在途状态迁移", B), T("机内 48–155 ms / 跨机约 1.3 s"), T("中"), T("同左（跨机受万兆网限制）")],
  ], { x: 0.6, y: 1.95, w: 12.1, colW: [2.6, 3.6, 1.2, 4.7], rowH: 0.62 });
  card(s, 0.6, 5.85, 12.1, 0.95, "E6EEF9");
  txt(s, "vLLM 已有同类技术：sleep 模式用 CUDA 虚拟内存（cuMem）整块卸载/恢复权重与 KV（实测 0.95 s / 0.74 s），但只能整块、KV 内容丢失、不能改变层归属——它为“让出整张卡”设计，不是为运行中改变显存用途。",
    { x: 0.85, y: 5.95, w: 11.7, h: 0.8, fontSize: 12, color: C.text1 });
  s.addNotes("见汇报稿第 16 页。");
}

// =============================================================== 17 VMM
{
  const s = content("实验：用 CUDA 虚拟内存只让出部分 KV 给新权重", "回答老师");
  table(s, [
    [H("步骤（6 GiB KV，让出 844 MiB = 3 个 Mamba 层）"), H("虚拟内存部分收缩"), H("现有做法：释放再申请")],
    [T("让出空间", B), T("解除映射 422 页（2 MiB/页）32.8 ms"), T("释放 + empty_cache 4.0 ms")],
    [T("给新权重分配显存", B), T("映射 422 页 21.1 ms"), T("申请 2.1 ms")],
    [T("拷贝权重（pinned → GPU）", B), T("36.2 ms"), T("72.8 ms")],
    [T("重建 KV", B), T("不需要"), T("申请并清零 8.8 ms")],
    [T("KV 基地址", B), T("不变", G), T("改变", R)],
    [T("留下的 KV 数据", B), T("完好", G), T("全部丢失", R)],
    [T("已录的 CUDA graph", B), T("直接回放，结果正确", G), T("失效，必须重录", R)],
  ], { x: 0.6, y: 1.3, w: 8.2, colW: [3.2, 2.6, 2.4], rowH: 0.56 });
  card(s, 9.1, 1.3, 3.6, 4.5);
  txt(s, "怎么做的", { x: 9.35, y: 1.45, w: 3.1, h: 0.4, fontSize: 15, bold: true });
  bullets(s, [
    "cuMemAddressReserve 为 KV 预留虚拟地址",
    "按 2 MiB 页 cuMemCreate + cuMemMap",
    "收缩：只 unmap/release 末尾页",
    "录一个读取保留 KV 的 graph，收缩后回放校验",
    "5 轮，取中位数；与 vAttention、kvcached 思路相同",
  ], { x: 9.35, y: 1.95, w: 3.1, h: 3.8, fontSize: 12 });
  takeaway(s, "显存侧可以做到：约 50 ms、地址不变、数据不丢、graph 不失效。剩下的是“层变了”带来的计算侧重录。", 6.45);
  s.addNotes("见汇报稿第 17 页。");
}

// =============================================================== 18 graph capture
{
  const s = content("实验：graph 重录 95% 是 Python 垃圾回收", "回答老师");
  const lb = ["2 个", "6 个", "10 个", "14 个"];
  s.addChart(pres.charts.BAR, [
    { name: "默认（14 层段）", labels: lb, values: [1021, 2972, 4837, 6835] },
    { name: "跳过 GC（14 层段）", labels: lb, values: [67, 167, 251, 340] },
    { name: "跳过 GC（42 层段）", labels: lb, values: [141, 351, 562, 790] },
  ], Object.assign(chartBase(), { x: 0.6, y: 1.25, w: 6.6, h: 5.0, barDir: "col", chartColors: [HEX.red, HEX.green, HEX.blue],
    showLegend: true, legendPos: "b", showTitle: true, title: "每段 graph 录制耗时（ms），横轴为 graph 个数", showValue: true,
    dataLabelPosition: "outEnd", dataLabelFormatCode: "0", valAxisMaxVal: 7500, valAxisMajorUnit: 1500, barGapWidthPct: 40 }));
  card(s, 7.5, 1.3, 5.2, 4.95);
  txt(s, "发现与做法", { x: 7.75, y: 1.45, w: 4.7, h: 0.4, fontSize: 15, bold: true });
  bullets(s, [
    "默认每个 graph 约 0.49 s，与层数无关（14 层、42 层一样）",
    "原因：torch.cuda.graph 每录一个都调 gc.collect() + empty_cache()，vLLM 再调一次 gc.collect()",
    "做法：仅在录 graph 期间跳过 gc.collect，录完恢复（插件开关）",
    "录制快 8–19 倍；graph 显存不变",
    "真实 PP=4 跨机：每段 20 个 graph，7–14 s → 0.41–0.42 s",
    "正确性：7/7 逐字一致",
    "修正：跳过 GC 后录制才与层数成正比（14 层 24 ms/个，42 层 55 ms/个）",
  ], { x: 7.75, y: 1.95, w: 4.7, h: 4.2, fontSize: 12 });
  takeaway(s, "换切分的最大代价从 10 秒级降到亚秒级；同样缩短每次服务启动。", 6.45);
  s.addNotes("见汇报稿第 18 页。");
}

// =============================================================== 19 code
pres.addSection({ title: "收尾" });
{
  const s = content("本周代码产出与 vLLM 修复", "收尾");
  table(s, [
    [H("模块 / 补丁"), H("作用")],
    [T("patches/nemotron_h_pp.py（已有）", { align: "left", bold: true }), T("NemotronH 流水线 4 个 bug：权重加载、层遍历、residual、中间结果工厂", { align: "left" })],
    [T("patches/spmd_fix.py（新）", { align: "left", bold: true }), T("bug 5：SPMD 路径漏传 finished_requests_ids 与 virtual_engine", { align: "left" })],
    [T("patches/capture_timing.py（新）", { align: "left", bold: true }), T("毫秒级记录每段 graph 录制；可选跳过录制期间的 GC", { align: "left" })],
    [T("pipeline_sim.py（新）", { align: "left", bold: true }), T("阶段计算模型拟合、V0 调度事件模拟、带容量检查的切分搜索", { align: "left" })],
    [T("comm_bench.py（新）", { align: "left", bold: true }), T("机内 / 跨机段间张量传输基准", { align: "left" })],
    [T("vmm_bench.py（新）", { align: "left", bold: true }), T("CUDA 虚拟内存部分收缩 vs 释放再申请", { align: "left" })],
    [T("graph_correctness.py（新）", { align: "left", bold: true }), T("两种配置输出对比：graph 上限 / SPMD / 跳过 GC；串行与并发", { align: "left" })],
    [T("reconfig_cost.py、capacity.py", { align: "left", bold: true }), T("单卡重配置成本基线、阶段显存估算", { align: "left" })],
  ], { x: 0.6, y: 1.25, w: 12.1, colW: [3.8, 8.3], rowH: 0.5 });
  takeaway(s, "代码与文档：github.com/nikiwang-125/hybrid-pp ，分支 modeling-20261111", 6.45);
  s.addNotes("见汇报稿第 19 页。");
}

// =============================================================== 20 limits
{
  const s = content("证据边界：哪些已确定，哪些还没有", "收尾");
  card(s, 0.6, 1.3, 5.95, 4.95);
  txt(s, "已确定（重复 + 正确性检查）", { x: 0.85, y: 1.45, w: 5.5, h: 0.4, fontSize: 15, bold: true, color: C.accent3 });
  bullets(s, [
    "单段计算模型对新切分误差 ≤5%",
    "graph 上限：8K +21%、16K +13%（n=3–4，t>6），输出不变",
    `SPMD + 修复：吞吐 +${D.spmd.thr1k}~${D.spmd.thr8k}%（n=3，t>10），7/7 一致`,
    "跳过 GC：重录 7–14 s → 0.4 s，7/7 一致",
    "开 graph 后默认均分最优或在噪声内",
    "VMM 部分收缩：地址、数据、graph 保持",
  ], { x: 0.85, y: 2.0, w: 5.5, h: 4.2, fontSize: 15, paraSpaceAfter: 10 });
  card(s, 6.75, 1.3, 5.95, 4.95);
  txt(s, "还没有 / 有局限", { x: 7.0, y: 1.45, w: 5.5, h: 0.4, fontSize: 15, bold: true, color: C.accent4 });
  bullets(s, [
    "TTFT 预测偏低 16–26%，原因未查清",
    "只测了一个模型、PP=4、并发 16、三种输入长度",
    "开 graph 后的切分对比每种只测 1 次",
    "VMM 是单卡合成实验，未接入 vLLM；未处理 Mamba 状态",
    "未实现端到端在线布局切换",
    "V1 引擎的对应能力未在本机验证",
  ], { x: 7.0, y: 2.0, w: 5.5, h: 4.2, fontSize: 15, paraSpaceAfter: 10 });
  s.addNotes("见汇报稿第 20 页。");
}

// =============================================================== 21 next
{
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "收尾" });
  txt(s, "结论与下一步", { x: 0.8, y: 0.55, w: 11.7, h: 0.8, fontSize: 32, bold: true, color: C.background1 });
  const cols = [
    ["回答了计划中的问题", [
      "瓶颈随负载变：默认配置下长输入卡在 graph 回退；修好后卡在主进程 CPU",
      "模型能预测量级并指出瓶颈；切分间差距小于其分辨率",
      "开 graph 后默认均分最优，切分无可测收益",
      "布局切换成本可从十秒级降到亚秒级（GC + VMM）",
      "优先级：先降主进程开销（SPMD），布局切换暂缓",
    ]],
    ["下一步（按优先级）", [
      "SPMD + graph 上限组合下重测切分与负载，重新拟合",
      "评估 vLLM V1：引擎与 worker 分进程、分段 graph",
      "给批次加编号，查清 TTFT 偏差",
      "把 VMM 部分收缩接入 vLLM 的 KV 与 Mamba 状态",
      "找切分收益更大的场景：更长上下文、更高并发、异构负载",
    ]],
  ];
  cols.forEach(([h, items], i) => {
    const x = 0.8 + i * 6.05;
    txt(s, h, { x, y: 1.6, w: 5.7, h: 0.5, fontSize: 19, bold: true, color: C.accent5 });
    txt(s, items.map((t, k) => ({ text: t, options: { bullet: true, breakLine: k < items.length - 1 } })),
      { x, y: 2.3, w: 5.7, h: 4.7, fontSize: 17, color: C.background1, paraSpaceAfter: 16 });
  });
  s.addNotes("见汇报稿第 21 页。");
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
