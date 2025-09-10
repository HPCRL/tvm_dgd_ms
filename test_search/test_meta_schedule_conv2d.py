# import tempfile
import os
import csv
import datetime
import numpy as np
import tvm
from tvm import meta_schedule as ms
from tvm.target import Target
from tvm import te

from typing import Tuple

from tvm import te, tir, topi
from tvm.target import Target

def conv2d_nchw(  # pylint: disable=invalid-name
    n: int,
    h: int,
    w: int,
    ci: int,
    co: int,
    kh: int,
    kw: int,
    stride: int,
    padding: int,
    dilation: int = 1,
    in_dtype: str = "float16",
    out_dtype: str = "float16",
) -> Tuple[te.Tensor, te.Tensor, te.Tensor]:
    x = te.placeholder((n, ci, h, w), name="X", dtype=in_dtype)
    w = te.placeholder((co, ci, kh, kw), name="W", dtype=in_dtype)
    y = topi.nn.conv2d_nchw(
        Input=x, Filter=w, stride=stride, padding=padding, dilation=dilation, out_dtype=out_dtype
    )
    return (x, w, y)

shapes_b1 = [
    (1, 3, 224, 224, 64, 3, 7, 7, 1, 2, 3, 1, 1),
    (1, 64, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1),
    (1, 64, 56, 56, 64, 64, 1, 1, 1, 1, 0, 1, 1),
    (1, 64, 56, 56, 128, 64, 3, 3, 1, 2, 1, 1, 1),
    (1, 64, 56, 56, 128, 64, 1, 1, 1, 2, 0, 1, 1),
    (1, 128, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1, 1),
    (1, 128, 28, 28, 256, 128, 3, 3, 1, 2, 1, 1, 1),
    (1, 128, 28, 28, 256, 128, 1, 1, 1, 2, 0, 1, 1),
    (1, 256, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1, 1),
    (1, 256, 14, 14, 512, 256, 3, 3, 1, 2, 1, 1, 1),
    (1, 256, 14, 14, 512, 256, 1, 1, 1, 2, 0, 1, 1),
    (1, 512, 7, 7, 512, 512, 3, 3, 1, 1, 1, 1, 1),
]

def make_prim_from_shape(shape, layout="nchw", dtype="float32"):
    # 形状解包（忽略占位的下划线位）
    N, C, H, W, K, _cpg, R, S, _kpg, stride, padding, dilation, groups = shape
    A, B, Conv = conv2d_nchw(N, H, W, C, K, R, S, stride, padding, dilation, dtype, dtype)
    return tvm.te.create_prim_func([A, B, Conv])


def prepare_unique_work_dir(base_work_dir: str) -> str:
    """Return a unique work directory path; avoid overwriting an existing one.

    If the provided base directory exists, append a timestamp suffix to create a
    unique directory name. Ensure the directory exists on return.
    """
    work_dir_path = base_work_dir
    if os.path.exists(work_dir_path):
        timestamp_suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        work_dir_path = f"{base_work_dir}_{timestamp_suffix}"
    os.makedirs(work_dir_path, exist_ok=True)
    return work_dir_path

def tune_and_benchmark(
    shape,
    target_str="nvidia/geforce-rtx-4090",
    work_dir_base="tmp_logs",
    max_trials_global: int = 1000,
    num_trials_per_iter: int = 64,
):
    prim = make_prim_from_shape(shape)
    target = Target(target_str)

    unique_work_dir = prepare_unique_work_dir(work_dir_base)

    database = ms.tir_integration.tune_tir(
        mod=prim,
        target=target,
        work_dir=unique_work_dir,
        max_trials_global=max_trials_global,  # 可按需调大
        num_trials_per_iter=num_trials_per_iter,
        runner=ms.runner.LocalRunner(
            evaluator_config=ms.runner.EvaluatorConfig(number=1, repeat=10, min_repeat_ms=10)
        ),
    )
    sch = ms.tir_integration.compile_tir(database, prim, target)
    if sch is None:
        return None, None

    mod = sch.mod
    func = tvm.build(mod, target=target)
    dev = tvm.device(target.kind.name, 0)

    # 构造输入
    N, C, H, W, K, _cpg, R, S, _kpg, stride, padding, dilation, groups = shape
    A_shape = (N, C, H, W)
    B_shape = (K, C, R, S)
    a_np = np.random.uniform(size=A_shape).astype("float32")
    b_np = np.random.uniform(size=B_shape).astype("float32")
    out_H = (H + 2*padding - ((R - 1)*dilation + 1)) // stride + 1
    out_W = (W + 2*padding - ((S - 1)*dilation + 1)) // stride + 1

    A_nd = tvm.nd.array(a_np, dev)
    B_nd = tvm.nd.array(b_np, dev)
    C_nd = tvm.nd.array(np.zeros((N, K, out_H, out_W), dtype="float32"), dev)

    evaluator = func.time_evaluator(func.entry_name, dev, number=10, repeat=10)
    mean_s = evaluator(A_nd, B_nd, C_nd).mean
    return func, mean_s * 1e3  # ms

if __name__ == "__main__":
    # 配置CSV结果目录与文件
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(results_dir, f"conv2d_results_{timestamp}.csv")

    # 固定配置
    layout = "nchw"
    in_dtype = "float32"
    out_dtype = "float32"
    simple_mode = False
    max_trials = 1000
    num_trials_iter = 64
    work_dir_base = "tmp_logs"

    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Timestamp",
                "Batch",
                "Layout",
                "In_dtype",
                "Out_dtype",
                "N",
                "C",
                "H",
                "W",
                "K",
                "R",
                "S",
                "Stride",
                "Padding",
                "Dilation",
                "Simple_mode",
                "Trials",
                "Cost_ms",
                "Status",
            ]
        )

        for idx, shape in enumerate(shapes_b1):
            N, C, H, W, K, _cpg, R, S, _kpg, stride, padding, dilation, groups = shape
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            try:
                _, t_ms = tune_and_benchmark(
                    shape,
                    work_dir_base=work_dir_base,
                    max_trials_global=max_trials,
                    num_trials_per_iter=num_trials_iter,
                )
                if t_ms is None:
                    status = "fail"
                    cost_ms = ""
                    print(f"[{idx}] no valid schedule")
                else:
                    status = "success"
                    cost_ms = f"{t_ms:.3f}"
                    print(f"[{idx}] latency: {t_ms:.3f} ms")
            except Exception as e:  # pylint: disable=broad-except
                status = "error"
                cost_ms = ""
                print(f"[{idx}] error during tuning/benchmark: {e}")

            writer.writerow(
                [
                    now_str,
                    N,
                    layout,
                    in_dtype,
                    out_dtype,
                    N,
                    C,
                    H,
                    W,
                    K,
                    R,
                    S,
                    stride,
                    padding,
                    dilation,
                    int(simple_mode),
                    max_trials,
                    cost_ms,
                    status,
                ]
            )