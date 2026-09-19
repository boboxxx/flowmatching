"""Validate complete measured author runs and write a conservative readable report."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    entries=[]
    for path in sorted(args.input.glob("*_crop*.summary.json")):
        content=json.loads(path.read_text())
        if content["status"]!="completed":
            continue
        csv_path=path.with_name(path.name.replace(".summary.json",".csv"))
        rows=list(csv.DictReader(csv_path.open()))
        for group in content["summary"]:
            select=[r for r in rows if float(r["snr_db"])==group["snr_db"]]
            assert len(select)==24*10, (path,"incomplete samples")
            assert len({r["image"] for r in select})==24
            assert len({(r["image"],r["draw"]) for r in select})==240
            for r in select:
                assert abs(float(r["psnr"])+10*math.log10(float(r["mse"])))<1e-5
                assert 0<=float(r["ms_ssim"])<=1
                assert float(r["lpips"])>=0
            for key in ["psnr","mse","ms_ssim","lpips","cbr_payload"]:
                mean=sum(float(r[key]) for r in select)/len(select)
                assert abs(mean-group[key])<1e-8
            entries.append(dict(model=content["checkpoint"]["name"],
                                objective=content["checkpoint"]["objective"],
                                crop=content["configuration"]["crop"],
                                checkpoint_sha256=content["checkpoint"]["sha256"],
                                csv_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(), **group))
    if not entries:
        raise ValueError("No complete benchmark files")
    report=["# 已发表论文基线：实测结果（2026-09-19）", "",
            "作者公开 checkpoint，主要复核 AWGN 10 dB，Kodak24 全部图像，每图 10 次独立信道噪声。",
            "若表中包含 0/5/15 dB，它们是固定 10 dB 权重的失配测试，不是分别重新训练。",
            "不是 10 个训练 seed；没有以训练种子标准差代替信道评估。",
            "这些结果是作者实现复核，不是与旧 Rayleigh H2/H3 的公平胜负比较。", ""]
    for crop in sorted({x["crop"] for x in entries}):
        report += ["## "+("原始分辨率" if crop==0 else f"中心 {crop}×{crop} 裁剪"), "",
                   "| 模型 / 训练目标 | SNR (dB) | payload CBR | CBR 含理想速率图 | PSNR ↑ | MSE ↓ | MS-SSIM ↑ | LPIPS ↓ |",
                   "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for e in entries:
            if e["crop"]==crop:
                report.append(f"| {e['model']} / {e['objective']} | {e['snr_db']:g} | {e['cbr_payload']:.5f} | "
                              f"{e['cbr_with_rate_map_capacity']:.5f} | {e['psnr']:.3f} | {e['mse']:.7f} | "
                              f"{e['ms_ssim']:.5f} | {e['lpips']:.5f} |")
        report += [""]
    report += ["## 解释边界", "",
               "- MSE 使用 RGB [0,1]；MS-SSIM 为统一标准五尺度；LPIPS 为 AlexNet v0.1。",
               "- SwinJSCC 的两种训练目标分别报告；MS-SSIM 模型不保证 LPIPS 也更好。",
               "- NTSCC 是公开 w/o z 变体，CBR 是实际选择的符号数。速率图按作者容量极限假设计费，未实现可靠数字编码。",
               "- 表中的 CBR 仍未计作者实现假定接收端可知的全局 latent scale；不能称为完整空口开销。",
               "- 作者预训练数据和网络规模不同；这些测量不能替代同数据、同信道、同反馈预算的主实验。",
               "- DeepJSCC-f 的训练/测试单列，未完成时不加入本表；旧简化适配器不能冒充论文复现。", "",
               "完整选择依据、论文 DOI、信道模型、指标与码率定义见 [protocol.md](protocol.md)。", ""]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text("\n".join(report))
    args.output.with_suffix(".json").write_text(json.dumps(entries,indent=2)+"\n")
    print(json.dumps(dict(validated_runs=len(entries),output=str(args.output))))

if __name__=="__main__":
    main()
