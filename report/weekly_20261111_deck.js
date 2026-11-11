// 本周汇报：混合模型流水线建模与方案搜索（2026-11-11）
// Build: NODE_PATH=<dir with pptxgenjs> APPLY_THEME=<path to apply_theme.js> node report/weekly_20261111_deck.js out.pptx report/weekly_20261111_capture.json
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.APPLY_THEME || "./apply_theme.js");

const OUT = process.argv[2] || "weekly.pptx";
const CAPTURE = process.argv[3] ? JSON.parse(require("fs").readFileSync(process.argv[3])) : null;

const HEX = {
  ink: "14213D", grey: "586174", light: "EDF1F5", white: "FFFFFF",
  blue: "2A78D6", amber: "E07A1F", green: "2E9E6B", red: "C8473D", mist: "9DB3D3",
};
const THEME = {
  name: "Pipeline Weekly",
  headFontFace: "Microsoft YaHei",
  bodyFontFace: "Microsoft YaHei",
  colors: {
    dk1: HEX.ink, lt1: HEX.white, dk2: HEX.grey, lt2: HEX.light,
    accent1: HEX.blue, accent2: HEX.amber, accent3: HEX.green, accent4: HEX.red,
    accent5: HEX.mist, accent6: HEX.ink, hlink: HEX.blue, folHlink: HEX.grey,
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "混合模型流水线建模与方案搜索 本周进展";
const C = pres.SchemeColor;
const W = 13.333;

// ---------------------------------------------------------------- layouts
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.3, w: 11.7, h: 1.6,
        fontSize: 40, bold: true, color: C.background1, valign: "bottom", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.1, w: 11.7, h: 1.2,
        fontSize: 18, color: C.accent5, valign: "top", align: "left" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.5, 0.6, 0.5, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.9,
        fontSize: 30, bold: true, color: C.text1, valign: "middle", align: "left", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 12.4, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: C.text2, align: "right" },
});

let n = 0;
function content(title, section) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  return s;
}
function txt(s, text, o) {
  s.addText(text, Object.assign({ isTextBox: true, margin: 0, color: C.text1, fontSize: 16,
    valign: "top" }, o));
}
function card(s, x, y, w, h, fill) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12,
    fill: { color: fill || C.background2 }, line: { type: "none" } });
}
function chip(s, x, y, label, fill, w) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: w || 1.1, h: 0.36, rectRadius: 0.18,
    fill: { color: fill }, line: { type: "none" } });
  txt(s, label, { x, y, w: w || 1.1, h: 0.36, fontSize: 12, bold: true, color: C.background1,
    align: "center", valign: "middle" });
}
const chartBase = () => ({
  catAxisLabelColor: HEX.grey, valAxisLabelColor: HEX.grey,
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", catAxisLabelFontSize: 12,
  valAxisLabelFontSize: 11, dataLabelFontFace: "+mn-lt", dataLabelFontSize: 11,
  dataLabelColor: HEX.ink, legendFontFace: "+mn-lt", legendFontSize: 12, legendColor: HEX.grey,
  titleFontFace: "+mn-lt", titleFontSize: 13, titleColor: HEX.grey,
  valGridLine: { color: "DDE3EA", size: 0.75 }, catGridLine: { style: "none" },
  valAxisLineShow: false,
});

// ================================================================ 1 title
pres.addSection({ title: "开场" });
{
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "开场" });
  s.addText("混合模型流水线建模与方案搜索", { placeholder: "title" });
  s.addText("本周进展  ·  Nemotron-H 9B  ·  双机 4 卡流水线（PP=4）  ·  2026-11-11", { placeholder: "body" });
  s.addNotes("本周目标：先弄清各类工作算多久、占多少显存、传数据多久，再预测不同切分的表现，挑候选上机器验证；并测一次换切分的成本。");
}

// ================================================================ 2 conclusion
{
  const s = content("一句话结论：长输入慢，主要是 graph 加速被关了", "开场");
  const stats = [
    ["≤ 5%", "单段耗时预测误差", "对没见过的切分也成立", C.accent1],
    ["+21%", "改一个启动参数后的提速", "8K 吞吐，重复 3 次；输出不变", C.accent3],
    ["7–14 s", "换一次切分的服务中断", "而换切分收益开 graph 后消失", C.accent4],
  ];
  stats.forEach(([big, label, sub, col], i) => {
    const x = 0.6 + i * 4.1;
    card(s, x, 1.6, 3.8, 3.3);
    txt(s, big, { x: x + 0.3, y: 1.9, w: 3.2, h: 1.3, fontSize: 54, bold: true, color: col });
    txt(s, label, { x: x + 0.3, y: 3.3, w: 3.2, h: 0.5, fontSize: 18, bold: true });
    txt(s, sub, { x: x + 0.3, y: 3.85, w: 3.2, h: 0.5, fontSize: 14, color: C.text2 });
  });
  txt(s, "建议：graph 长度上限设为最大上下文；在线换切分暂缓，优先降低每步开销、改进调度。", {
    x: 0.6, y: 5.5, w: 12.1, h: 0.6, fontSize: 20, bold: true, color: C.text1 });
  s.addNotes("三个数字串起整周：模型够准；长输入真正的瓶颈是 vLLM 超过 8192 token 关掉 CUDA graph，改 --max-seq-len-to-capture 后 8K 吞吐 +21%、16K +13%（各重复 3 次，t>6），输出与原来一致；此时默认均分就是最好的切分，换切分的 5–7% 收益消失，而换一次要中断 7–14 s。");
}

// ================================================================ 3 what we did
pres.addSection({ title: "做了什么" });
{
  const s = content("本周做了什么：从测量到验证跑通一遍", "做了什么");
  const steps = [
    ["测量", "统一口径", "部分", C.accent2],
    ["计算", "每段算多久", "初版", C.accent1],
    ["显存", "装不装得下", "完成", C.accent3],
    ["通信", "传数据多久", "完成", C.accent3],
    ["模拟", "整条线怎么跑", "初版", C.accent1],
    ["搜索", "挑切分", "完成", C.accent3],
    ["验证", "上机器实测", "完成", C.accent3],
  ];
  const w = 1.6, gap = 0.15, x0 = 0.6;
  steps.forEach(([name, q, st, col], i) => {
    const x = x0 + i * (w + gap);
    s.addShape(pres.shapes.OVAL, { x: x + 0.45, y: 1.7, w: 0.7, h: 0.7,
      fill: { color: C.text1 }, line: { type: "none" } });
    txt(s, String(i + 1), { x: x + 0.45, y: 1.7, w: 0.7, h: 0.7, fontSize: 20, bold: true,
      color: C.background1, align: "center", valign: "middle" });
    txt(s, name, { x, y: 2.6, w, h: 0.45, fontSize: 20, bold: true, align: "center" });
    txt(s, q, { x, y: 3.1, w, h: 0.4, fontSize: 13, color: C.text2, align: "center" });
    chip(s, x + 0.25, 3.65, st, col);
    if (i < steps.length - 1)
      s.addShape(pres.shapes.LINE, { x: x + 1.2, y: 2.05, w: w + gap - 0.75, h: 0,
        line: { color: HEX.mist, width: 1.5, endArrowType: "triangle" } });
  });
  card(s, 0.6, 4.6, 12.1, 1.25);
  txt(s, [
    { text: "实验规模：", options: { bold: true } },
    { text: "7 种切分 × 3 种输入长度（1K / 8K / 16K），关键对比重复 3–4 次；共约 40 次四卡实测", options: { breakLine: true } },
    { text: "附带：", options: { bold: true } },
    { text: "显存可变性源码核查、换切分成本基线、通信基准", options: {} },
  ], { x: 0.9, y: 4.8, w: 11.5, h: 0.9, fontSize: 16, paraSpaceAfter: 8 });
  s.addNotes("“初版”表示能用但有明确缺口；“部分”表示测量口径还缺 micro-batch 标识。");
}

// ================================================================ 4 compute model
pres.addSection({ title: "建模" });
{
  const s = content("计算模型：数一数每段有几个什么层，就能估出耗时", "建模");
  card(s, 0.6, 1.5, 5.4, 5.0);
  txt(s, "怎么做", { x: 0.9, y: 1.75, w: 4.8, h: 0.5, fontSize: 20, bold: true });
  txt(s, [
    { text: "模型由三种层组成：Mamba、attention、MLP", options: { bullet: true, breakLine: true } },
    { text: "从实测反推每种层要花多久、数据多了慢多少", options: { bullet: true, breakLine: true } },
    { text: "新切分：按层数相加即可预测，不用上机器", options: { bullet: true, breakLine: true } },
    { text: "检验：一个切分（15,13,15,13）事先藏起来，不参与建模，专门用来考", options: { bullet: true } },
  ], { x: 0.9, y: 2.35, w: 4.8, h: 3.0, fontSize: 16, paraSpaceAfter: 10 });
  txt(s, "未做到：单个层级别的预测（目前只到整段）", { x: 0.9, y: 5.65, w: 4.8, h: 0.6,
    fontSize: 14, color: C.accent4, bold: true });
  s.addChart(pres.charts.BAR, [{ name: "误差", labels: ["生成（graph 模式）", "生成（eager 模式）", "读输入（prefill）"],
    values: [1.1, 4.9, 0.7] }], Object.assign(chartBase(), {
    x: 6.4, y: 1.5, w: 6.3, h: 5.0, barDir: "col", chartColors: [HEX.blue],
    showTitle: true, title: "藏起来的切分上，每段耗时预测误差（%）",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0.0"%"',
    valAxisMinVal: 0, valAxisMaxVal: 6, valAxisMajorUnit: 2, showLegend: false, barGapWidthPct: 80,
  }));
  s.addNotes("每段耗时 = 各类层数 × 各自成本 + 与 batch、token 数、上下文长度相关的项。graph/eager 分开拟合，因为两者差 2–4 倍。");
}

// ================================================================ 5 memory
{
  const s = content("显存：权重估得很准，KV 缓存启动后就不能改大小", "建模");
  const stages = ["均分 段1", "段2", "段3", "段4", "18,10,10,18 段1", "段2", "段3", "段4"];
  s.addChart(pres.charts.BAR, [
    { name: "权重", labels: stages, values: [5.96, 5.62, 5.62, 5.95, 6.86, 4.72, 4.70, 6.86] },
    { name: "运行时临时空间", labels: stages, values: [4.08, 4.06, 4.06, 4.14, 4.16, 3.84, 3.90, 4.16] },
    { name: "留给 KV 缓存", labels: stages, values: [11.15, 11.52, 11.52, 11.11, 10.17, 12.64, 12.59, 10.17] },
  ], Object.assign(chartBase(), {
    x: 0.6, y: 1.45, w: 7.4, h: 5.1, barDir: "col", barGrouping: "stacked",
    chartColors: [HEX.ink, HEX.mist, HEX.blue], showLegend: true, legendPos: "b",
    showTitle: true, title: "每张卡显存去向（GiB）", showValue: true, dataLabelPosition: "ctr",
    dataLabelColor: HEX.white, dataLabelFormatCode: "0.0", valAxisMaxVal: 24, valAxisMajorUnit: 4,
    catAxisLabelFontSize: 10, barGapWidthPct: 40,
  }));
  const pts = [
    ["估算 vs 实测", "权重误差 < 0.01 GiB", C.accent3],
    ["哪张卡先满", "两头放太多层，KV 空间被挤小；整体按最小的那张卡算", C.accent2],
    ["能不能动态调", "KV 缓存启动时一次性划定，改大小 = 释放再申请，服务要停", C.accent4],
  ];
  pts.forEach(([h, b, col], i) => {
    const y = 1.55 + i * 1.7;
    card(s, 8.3, y, 4.4, 1.5);
    s.addShape(pres.shapes.OVAL, { x: 8.5, y: y + 0.25, w: 0.28, h: 0.28, fill: { color: col }, line: { type: "none" } });
    txt(s, h, { x: 8.95, y: y + 0.18, w: 3.6, h: 0.4, fontSize: 16, bold: true });
    txt(s, b, { x: 8.95, y: y + 0.62, w: 3.6, h: 0.8, fontSize: 13, color: C.text2 });
  });
  s.addNotes("运行时临时空间约 4.1 GiB/卡，由启动时按 16384 token 的 prefill 预估。Mamba 状态按最大并发数预分配，与实际请求数无关，sleep 也释放不了。当前负载下 6545 种切分都装得下。");
}

// ================================================================ 6 comm
{
  const s = content("通信：跨机器比机器内慢 16 倍", "建模");
  s.addChart(pres.charts.BAR, [
    { name: "同一台机器内", labels: ["32 token\n（生成）", "512 token", "2048 token", "8192 token\n（读长输入）"], values: [0.09, 0.57, 2.1, 8.2] },
    { name: "跨两台机器", labels: ["32 token\n（生成）", "512 token", "2048 token", "8192 token\n（读长输入）"], values: [0.63, 7.9, 31.3, 124.9] },
  ], Object.assign(chartBase(), {
    x: 0.6, y: 1.45, w: 7.6, h: 5.1, barDir: "col", chartColors: [HEX.mist, HEX.amber],
    showLegend: true, legendPos: "b", showTitle: true, title: "传一次中间结果的耗时（毫秒）",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", valAxisMaxVal: 140,
    valAxisMajorUnit: 20, barGapWidthPct: 60,
  }));
  card(s, 8.6, 1.55, 4.1, 2.2);
  txt(s, "18.6 vs 1.18", { x: 8.9, y: 1.75, w: 3.6, h: 0.8, fontSize: 32, bold: true, color: C.accent2 });
  txt(s, "GB/s：机器内 vs 跨机器\n跨机走普通万兆网线，已接近跑满", { x: 8.9, y: 2.6, w: 3.6, h: 1.0, fontSize: 13, color: C.text2 });
  card(s, 8.6, 4.0, 4.1, 2.5);
  txt(s, "影响", { x: 8.9, y: 4.2, w: 3.6, h: 0.4, fontSize: 16, bold: true });
  txt(s, [
    { text: "生成阶段：每次 < 1 ms，可以忽略", options: { bullet: true, breakLine: true } },
    { text: "读长输入：2048 token 跨机 31 ms，和一段的计算差不多 → 切点位置要算进去", options: { bullet: true } },
  ], { x: 8.9, y: 4.65, w: 3.6, h: 1.8, fontSize: 13, color: C.text1, paraSpaceAfter: 6 });
  s.addNotes("默认映射：段1、2 在 slave21，段3、4 在 slave20，只有段2→段3 跨机。单向计时与往返计时一致，说明孤立测量没有混入等待。");
}

// ================================================================ 7 simulator
{
  const s = content("流水线模拟器：吞吐和每字间隔估得准，首字时间偏低", "建模");
  const hdr = { bold: true, color: HEX.white, fill: { color: HEX.ink }, fontSize: 14, align: "center", valign: "middle" };
  const cell = (t, o) => ({ text: t, options: Object.assign({ fontSize: 15, align: "center", valign: "middle", color: HEX.ink }, o || {}) });
  const good = { color: HEX.green, bold: true }, bad = { color: HEX.red, bold: true };
  s.addTable([
    [{ text: "藏起来的切分", options: hdr }, { text: "吞吐", options: hdr }, { text: "每字间隔 TPOT", options: hdr }, { text: "首字时间 TTFT", options: hdr }],
    [cell("短输入 1K", { bold: true }), cell("−1.5%", good), cell("+3.4%", good), cell("−7.3%")],
    [cell("长输入 8K", { bold: true }), cell("+8.4%", good), cell("−2.2%", good), cell("−26%", bad)],
  ], { x: 0.6, y: 1.6, w: 7.4, colW: [1.9, 1.6, 1.95, 1.95], rowH: 0.75,
    border: { type: "solid", color: "DDE3EA", pt: 1 }, fill: { color: HEX.white } });
  txt(s, "数字 = 预测相对实测的误差", { x: 0.6, y: 4.0, w: 7.4, h: 0.4, fontSize: 12, color: C.text2 });
  card(s, 0.6, 4.6, 7.4, 1.9);
  txt(s, "模拟了什么", { x: 0.9, y: 4.8, w: 6.8, h: 0.4, fontSize: 16, bold: true });
  txt(s, "vLLM 的 4 个虚拟引擎轮流往流水线里送批次；每张卡一次只算一批；同一请求要等上一个字出来才能算下一个；卡与卡之间按通信模型传数据。",
    { x: 0.9, y: 5.25, w: 6.8, h: 1.2, fontSize: 13, color: C.text2 });
  const issues = [
    ["分不清谁更好", "各切分实测只差 5–8%，和模型误差同量级；预测把差距放大了约 2 倍"],
    ["首字时间偏低", "读输入阶段有开销没建模；需要给每个批次加编号才能拆开“等”和“传”"],
    ["不支持混合步", "16K 以上会把读输入和生成混在一步，模拟器还不支持"],
  ];
  issues.forEach(([h, b], i) => {
    const y = 1.6 + i * 1.65;
    card(s, 8.4, y, 4.3, 1.45);
    chip(s, 8.6, y + 0.18, "待解决", C.accent4, 1.0);
    txt(s, h, { x: 9.75, y: y + 0.15, w: 2.8, h: 0.4, fontSize: 15, bold: true });
    txt(s, b, { x: 8.6, y: y + 0.65, w: 3.95, h: 0.75, fontSize: 12, color: C.text2 });
  });
  s.addNotes("唯一校准常数：每段每步约 8 ms 的非 GPU 开销（收发、元数据、调度），用训练切分的 TPOT 拟合得到。");
}

// ================================================================ 8 validation
pres.addSection({ title: "验证" });
{
  const s = content("实测（vLLM 默认配置）：推荐切分只在长输入占优", "验证");
  const labels = ["短输入 1K", "长输入 8K", "超长 16K"];
  s.addChart(pres.charts.BAR, [
    { name: "默认均分 14,14,14,14", labels, values: [60.8, 195.6, 305.2] },
    { name: "模型推荐 7,17,18,14", labels, values: [65.2, 181.1, 286.3] },
  ], Object.assign(chartBase(), {
    x: 0.6, y: 1.45, w: 7.4, h: 5.1, barDir: "col", chartColors: [HEX.mist, HEX.amber],
    showLegend: true, legendPos: "b", showTitle: true, title: "每个字的生成间隔 TPOT（毫秒，越低越好）",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0", valAxisMaxVal: 350,
    valAxisMajorUnit: 50, barGapWidthPct: 60,
  }));
  const rows = [
    ["短输入", "推荐方案更差", "吞吐 −5.8%，每字慢 7.3%", C.accent4, "非常确定"],
    ["长输入 8K", "推荐方案更好", "每字快 7.5%，吞吐约 +5%", C.accent3, "确定"],
    ["超长 16K", "推荐方案更好", "每字快 6%，吞吐 +3%", C.accent3, "只测 1 次"],
  ];
  rows.forEach(([h, v, b, col, conf], i) => {
    const y = 1.55 + i * 1.7;
    card(s, 8.3, y, 4.4, 1.5);
    txt(s, h, { x: 8.55, y: y + 0.15, w: 2.0, h: 0.4, fontSize: 14, color: C.text2 });
    chip(s, 11.3, y + 0.15, conf, HEX.grey, 1.2);
    txt(s, v, { x: 8.55, y: y + 0.55, w: 3.9, h: 0.4, fontSize: 17, bold: true, color: col });
    txt(s, b, { x: 8.55, y: y + 0.98, w: 3.9, h: 0.4, fontSize: 13 });
  });
  s.addNotes("默认重复 4 次、推荐重复 3 次，Welch t 检验：短输入吞吐 t=−20，长输入 TPOT t=−5.6，长输入吞吐 t=2.2（仅倾向）。长输入单次吞吐波动约 3%，单次比较不可靠。");
}

// ================================================================ 9 why
{
  const s = content("为什么：长输入时第 1 张卡特别慢", "验证");
  s.addChart(pres.charts.BAR, [
    { name: "生成阶段 GPU 忙碌时间", labels: ["段 1", "段 2", "段 3", "段 4"], values: [41.0, 14.6, 14.8, 16.0] },
  ], Object.assign(chartBase(), {
    x: 0.6, y: 1.45, w: 6.2, h: 5.1, barDir: "col", chartColors: [HEX.amber, HEX.mist, HEX.mist, HEX.mist],
    invertedColors: [HEX.amber], showLegend: false, showTitle: true,
    title: "默认均分（每段都是 14 层），16K 输入，单位：秒",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", valAxisMaxVal: 50,
    valAxisMajorUnit: 10, barGapWidthPct: 70,
  }));
  const pts = [
    ["1", "超过 8192 token，vLLM 关掉 CUDA graph 加速", "退回 eager 模式：每个小计算都要 CPU 现场下命令，每段慢 2–4 倍"],
    ["2", "第 1 张卡和 vLLM 的总调度挤在一起抢 CPU", "eager 最依赖 CPU，所以第 1 段最吃亏；开 graph 后段 1 不再慢（7.4 s vs 其他 6.5–8.1 s）"],
    ["3", "所以“第 1 段少放层”只在长输入有效", "短输入走 graph，CPU 开销小，反而让中间段变重"],
  ];
  pts.forEach(([k, h, b], i) => {
    const y = 1.55 + i * 1.7;
    s.addShape(pres.shapes.OVAL, { x: 7.2, y: y + 0.1, w: 0.55, h: 0.55, fill: { color: C.text1 }, line: { type: "none" } });
    txt(s, k, { x: 7.2, y: y + 0.1, w: 0.55, h: 0.55, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle" });
    txt(s, h, { x: 7.95, y: y + 0.05, w: 4.75, h: 0.7, fontSize: 16, bold: true });
    txt(s, b, { x: 7.95, y: y + 0.8, w: 4.75, h: 0.7, fontSize: 13, color: C.text2 });
  });
  s.addNotes("同样 14 层，段 1 的生成耗时是其他段的 2.8 倍；推荐切分里段 1 只有 7 层，仍比 14 层的段 4 慢一倍。根因在 vLLM 源码：CUDA graph 录制时地址表按 8192 token 定长分配，超长请求装不下，只能整步退回 eager。");
}

// ================================================================ 10 reconfig cost
{
  const s = content("换一次切分要多久：大头是重建 CUDA graph", "验证");
  const labels = ["释放 / 重建 KV 缓存", "搬 3 层权重（内存→显存）", "搬正在服务的请求状态", "跨机器搬 1.5 GiB 状态", "重建 CUDA graph", "整个服务重启"];
  s.addChart(pres.charts.BAR, [{ name: "耗时", labels, values: [6, 47, 155, 1300, 10000, 60000] }],
    Object.assign(chartBase(), {
      x: 0.6, y: 1.45, w: 8.2, h: 5.1, barDir: "bar", chartColors: [HEX.mist, HEX.mist, HEX.mist, HEX.mist, HEX.red, HEX.ink],
      showLegend: false, showTitle: true, title: "耗时（毫秒，对数刻度）", valAxisLogScaleBase: 10,
      valAxisMinVal: 1, valAxisMaxVal: 100000, showValue: true, dataLabelPosition: "outEnd",
      dataLabelFormatCode: "#,##0", catAxisOrientation: "maxMin", barGapWidthPct: 50,
    }));
  card(s, 9.2, 1.55, 3.5, 2.4);
  txt(s, "7–14 s", { x: 9.5, y: 1.75, w: 3.0, h: 0.9, fontSize: 40, bold: true, color: C.accent4 });
  txt(s, "每张卡要为 4 个虚拟引擎各录 5 套 graph，共 20 个；KV 一换地址，全部作废", { x: 9.5, y: 2.65, w: 3.0, h: 1.2, fontSize: 13, color: C.text2 });
  card(s, 9.2, 4.15, 3.5, 2.35);
  txt(s, "算一笔账", { x: 9.5, y: 4.35, w: 3.0, h: 0.4, fontSize: 16, bold: true });
  txt(s, "收益 5–7%，要省回 10 秒左右的中断，负载得稳定好几分钟才划算", { x: 9.5, y: 4.8, w: 3.0, h: 1.5, fontSize: 13, color: C.text1 });
  s.addNotes("单卡实测：移 3 个 attention 层 / Mamba 层 / MLP 层分别测；sleep 模式 0.95 s 睡、0.74 s 醒，但 KV 内容丢失、约 1 GiB 释放不掉，且不能改变层归属。graph 重建时间取自 16 次 PP=4 启动日志。");
}

// ================================================================ 11 capture experiment
{
  const s = content("调大 graph 上限：长输入吞吐 +13~21%", "验证");
  card(s, 0.6, 1.5, 6.0, 5.0);
  txt(s, "做法与结果", { x: 0.9, y: 1.75, w: 5.4, h: 0.4, fontSize: 18, bold: true });
  txt(s, [
    { text: "模型 56 层里只有 4 层 attention 需要那张定长地址表", options: { bullet: true, breakLine: true } },
    { text: "但 graph 是整段录的：4 层超限，另外 52 层也跟着退回 eager", options: { bullet: true, breakLine: true } },
    { text: "把 --max-seq-len-to-capture 设为最大上下文：长输入全部走 graph", options: { bullet: true, breakLine: true } },
    { text: "8K：吞吐 +21%，每字快 24%；16K：吞吐 +13%（各 3 次，显著）", options: { bullet: true, breakLine: true } },
    { text: "录制时间、显存、KV 空间不变；输出 6/7 逐字一致，另 1 条只差一个词，答案相同", options: { bullet: true } },
  ], { x: 0.9, y: 2.25, w: 5.4, h: 3.0, fontSize: 14, paraSpaceAfter: 8 });
  txt(s, "开 graph 后默认均分最好，推荐切分反而差 9–11%：之前的优势只是在躲 eager 开销", {
    x: 0.9, y: 5.35, w: 5.4, h: 1.0, fontSize: 15, bold: true, color: C.accent1 });
  if (CAPTURE) {
    const labels = CAPTURE.labels;
    s.addChart(pres.charts.BAR, [
      { name: "默认上限 8192", labels, values: CAPTURE.before },
      { name: "上限放到最大", labels, values: CAPTURE.after },
    ], Object.assign(chartBase(), {
      x: 7.0, y: 1.45, w: 5.7, h: 5.1, barDir: "col", chartColors: [HEX.mist, HEX.blue],
      showLegend: true, legendPos: "b", showTitle: true, title: CAPTURE.title,
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0", barGapWidthPct: 60,
    }));
  } else {
    card(s, 7.0, 1.5, 5.7, 5.0, HEX.white);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.0, y: 1.5, w: 5.7, h: 5.0, rectRadius: 0.12,
      fill: { color: HEX.white }, line: { color: HEX.mist, width: 1.5, dashType: "dash" } });
    txt(s, "实测进行中", { x: 7.3, y: 3.3, w: 5.1, h: 0.6, fontSize: 24, bold: true, color: C.text2, align: "center" });
    txt(s, "8K / 16K 输入 × 默认与推荐切分", { x: 7.3, y: 4.0, w: 5.1, h: 0.5, fontSize: 14, color: C.text2, align: "center" });
  }
  s.addNotes("默认均分各重复 3–4 次：8K 吞吐 58.2→70.7（t=6.3），TPOT 196→148 ms；16K 42.5→48.0（t=12.1）；TTFT 不变（读输入本来不走 graph）。推荐切分各 1 次。正确性：7 条提示（含 4 条 9.7K–15K token 检索题）greedy 400 token，6 条逐字一致，1 条在思考文字中一词不同（graph/eager 浮点差异），检索正确率两者相同。16K 提升较小是因为开了 chunked prefill，读输入与生成混合的步骤仍不能用 graph。源码依据：worker/model_runner.py 中 graph_block_tables 形状为 (最大 batch, max_seq_len_to_capture / 16)；_use_captured_graph 要求 batch 内最长序列 ≤ 该值。");
}

// ================================================================ 12 next
pres.addSection({ title: "结论" });
{
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "结论" });
  txt(s, "结论与下一步", { x: 0.8, y: 0.6, w: 11.7, h: 0.9, fontSize: 34, bold: true, color: C.background1 });
  const cols = [
    ["回答了什么", [
      "长输入慢的主因是 graph 被关；改参数后吞吐 +13~21%，输出不变",
      "单段预测误差 ≤5%；但切分间的差距小于模型能分辨的范围",
      "开 graph 后默认均分最好，换切分没有收益，却要中断 7–14 s",
    ]],
    ["下一步（按优先级）", [
      "上限放大后重做 7 切分对比，重新拟合模型",
      "降低每步约 8 ms 的非 GPU 开销（收发、调度）",
      "给批次加编号，查清首字时间偏差",
      "读输入与生成混合的步骤也要能用 graph / 能模拟",
    ]],
  ];
  cols.forEach(([h, items], i) => {
    const x = 0.8 + i * 6.0;
    txt(s, h, { x, y: 1.8, w: 5.6, h: 0.5, fontSize: 20, bold: true, color: C.accent5 });
    txt(s, items.map((t, k) => ({ text: t, options: { bullet: true, breakLine: k < items.length - 1 } })),
      { x, y: 2.45, w: 5.6, h: 3.8, fontSize: 16, color: C.background1, paraSpaceAfter: 12 });
  });
  s.addNotes("暂不做在线布局切换；先做开销和调度。代码与文档：github.com/nikiwang-125/hybrid-pp 分支 modeling-20261111。");
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
