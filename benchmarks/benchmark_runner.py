from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from matplotlib import pyplot as plt

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore

try:
    from .resource_monitor import ResourceMonitor
except ImportError:  # pragma: no cover
    from resource_monitor import ResourceMonitor  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = ROOT / "example"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import Materials  # type: ignore
import scipy.io as sio  # type: ignore
import torcwa  # type: ignore


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but not available")
        return torch.device("cuda")
    if requested != "auto":
        raise ValueError(f"Unknown device option: {requested}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@contextmanager
def working_directory(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def rcwa_warmup(device: torch.device, geo_dtype: torch.dtype, sim_dtype: torch.dtype) -> None:
    if not torch.cuda.is_available():
        return
    with working_directory(EXAMPLE_DIR):
        lamb = torch.tensor(600.0, dtype=geo_dtype, device=device)
        torcwa.rcwa_geo.dtype = geo_dtype
        torcwa.rcwa_geo.device = device
        torcwa.rcwa_geo.Lx = 200.0
        torcwa.rcwa_geo.Ly = 200.0
        torcwa.rcwa_geo.nx = 64
        torcwa.rcwa_geo.ny = 64
        torcwa.rcwa_geo.grid()
        torcwa.rcwa_geo.edge_sharpness = 1000.0
        layer = torcwa.rcwa_geo.square(W=50.0, Cx=100.0, Cy=100.0)
        silicon_eps = Materials.aSiH.apply(lamb) ** 2
        layer_eps = layer * silicon_eps + (1.0 - layer)
        sim = torcwa.rcwa(freq=1.0 / lamb, order=[3, 3], L=[200.0, 200.0], dtype=sim_dtype, device=device)
        sim.add_input_layer(eps=1.0)
        sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
        sim.add_layer(thickness=100.0, eps=layer_eps)
        sim.solve_global_smatrix()
        sim.S_parameters(orders=[0, 0], direction="forward", port="transmission", polarization="xx", ref_order=[0, 0])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        del sim
        torch.cuda.empty_cache()


def setup_example2(device: torch.device, geo_dtype: torch.dtype, sim_dtype: torch.dtype) -> Dict[str, object]:
    with working_directory(EXAMPLE_DIR):
        lamb0 = torch.tensor(532.0, dtype=geo_dtype, device=device)
        inc_ang = 15.0 * (np.pi / 180.0)
        azi_ang = 0.0 * (np.pi / 180.0)
        substrate_eps = 1.46 ** 2
        silicon_eps = Materials.aSiH.apply(lamb0) ** 2
        L = [300.0, 300.0]
        torcwa.rcwa_geo.dtype = geo_dtype
        torcwa.rcwa_geo.device = device
        torcwa.rcwa_geo.Lx = L[0]
        torcwa.rcwa_geo.Ly = L[1]
        torcwa.rcwa_geo.nx = 300
        torcwa.rcwa_geo.ny = 300
        torcwa.rcwa_geo.grid()
        torcwa.rcwa_geo.edge_sharpness = 1000.0
        layer0_geometry = torcwa.rcwa_geo.square(W=120.0, Cx=L[0] / 2.0, Cy=L[1] / 2.0)
        layer0_eps = layer0_geometry * silicon_eps + (1.0 - layer0_geometry)
        layer0_thickness = 300.0
        z = torch.linspace(-500.0, 1500.0, 501, device=device)
    x_axis = torcwa.rcwa_geo.x.cpu()
    y_axis = torcwa.rcwa_geo.y.cpu()
    z_axis = z.cpu()
    return {
        "device": device,
        "geo_dtype": geo_dtype,
        "sim_dtype": sim_dtype,
        "lamb0": lamb0,
        "inc_ang": float(inc_ang),
        "azi_ang": float(azi_ang),
        "substrate_eps": float(substrate_eps),
        "layer0_geometry": layer0_geometry,
        "layer0_eps": layer0_eps,
        "layer0_thickness": float(layer0_thickness),
        "L": L,
        "z": z,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "z_axis": z_axis,
    }


def setup_example6(device: torch.device, geo_dtype: torch.dtype, sim_dtype: torch.dtype) -> Dict[str, object]:
    with working_directory(EXAMPLE_DIR):
        lamb0 = torch.tensor(532.0, dtype=geo_dtype, device=device)
        inc_ang = 0.0
        azi_ang = 0.0
        substrate_eps = 1.46 ** 2
        silicon_eps = Materials.aSiH.apply(lamb0) ** 2
        L = [700.0, 300.0]
        torcwa.rcwa_geo.dtype = geo_dtype
        torcwa.rcwa_geo.device = device
        torcwa.rcwa_geo.Lx = L[0]
        torcwa.rcwa_geo.Ly = L[1]
        torcwa.rcwa_geo.nx = 700
        torcwa.rcwa_geo.ny = 300
        torcwa.rcwa_geo.grid()
        torcwa.rcwa_geo.edge_sharpness = 1000.0
        layer0_thickness = 300.0
    return {
        "device": device,
        "geo_dtype": geo_dtype,
        "sim_dtype": sim_dtype,
        "lamb0": lamb0,
        "inc_ang": float(inc_ang),
        "azi_ang": float(azi_ang),
        "substrate_eps": float(substrate_eps),
        "silicon_eps": silicon_eps,
        "layer0_thickness": float(layer0_thickness),
        "L": L,
    }


def run_example2_order_sweep(
    monitor: ResourceMonitor,
    config: Dict[str, object],
    orders: Sequence[int],
    output_dir: Path,
    store_transfer: bool,
    field_probe_order: Optional[int],
    store_field: bool,
    progress: bool,
) -> List[Dict[str, object]]:
    device = config["device"]
    sim_dtype = config["sim_dtype"]
    results: List[Dict[str, object]] = []
    monitor.mark("example2_order_sweep_start", metadata={"order_min": int(min(orders)), "order_max": int(max(orders))})

    progress_bar = None
    if progress and tqdm is not None:
        progress_bar = tqdm(orders, desc="Order Sweep", unit="order")
        order_iter: Iterable[int] = progress_bar
    else:
        order_iter = orders

    try:
        for order_N in order_iter:
            order_meta = {"order": int(order_N)}
            monitor.mark("example2_order_setup", metadata=order_meta)
            sim = torcwa.rcwa(
                freq=1.0 / config["lamb0"],
                order=[order_N, order_N],
                L=config["L"],
                dtype=sim_dtype,
                device=device,
            )
            sim.add_input_layer(eps=config["substrate_eps"])
            sim.set_incident_angle(inc_ang=config["inc_ang"], azi_ang=config["azi_ang"])
            sim.add_layer(thickness=config["layer0_thickness"], eps=config["layer0_eps"])

            monitor.mark("example2_order_smatrix_start", metadata=order_meta)
            sim.solve_global_smatrix()
            monitor.mark("example2_order_smatrix_end", metadata=order_meta, sync_cuda=True)

            txx = sim.S_parameters(orders=[0, 0], direction="forward", port="transmission", polarization="xx", ref_order=[0, 0])
            tyy = sim.S_parameters(orders=[0, 0], direction="forward", port="transmission", polarization="yy", ref_order=[0, 0])
            monitor.mark("example2_order_sparams", metadata=order_meta, sync_cuda=True)

            results.append({
                "order": int(order_N),
                "txx": complex(txx.detach().cpu().item()),
                "tyy": complex(tyy.detach().cpu().item()),
            })

            if progress_bar is not None:
                progress_bar.set_postfix(order=order_N)

            if store_transfer:
                out_path = output_dir / f"example2_order_{order_N:02d}_Sparams.mat"
                sio.savemat(str(out_path), {"txx": txx.cpu().numpy(), "tyy": tyy.cpu().numpy()})

            if field_probe_order is not None and order_N == field_probe_order:
                monitor.mark("example2_field_probe_start", metadata=order_meta)
                [Ex, Ey, Ez], [Hx, Hy, Hz] = sim.field_xz(torcwa.rcwa_geo.x, config["z"], config["L"][1] / 2.0)
                Enorm = torch.sqrt(torch.abs(Ex) ** 2 + torch.abs(Ey) ** 2 + torch.abs(Ez) ** 2)
                Hnorm = torch.sqrt(torch.abs(Hx) ** 2 + torch.abs(Hy) ** 2 + torch.abs(Hz) ** 2)
                monitor.mark("example2_field_probe_end", metadata=order_meta, sync_cuda=True)
                if store_field:
                    np.savez(
                        output_dir / f"example2_order_{order_N:02d}_field_probe.npz",
                        x=config["x_axis"],
                        z=config["z_axis"],
                        Enorm=Enorm.detach().cpu().numpy(),
                        Hnorm=Hnorm.detach().cpu().numpy(),
                    )
            del sim
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if progress_bar is not None:
            progress_bar.close()

    monitor.mark("example2_order_sweep_end", metadata={"count": len(orders)})
    return results



def run_example2_wavelength_sweep(
    monitor: ResourceMonitor,
    config: Dict[str, object],
    wavelengths_nm: Sequence[float],
    order_N: int,
    output_dir: Path,
    store_transfer: bool,
    progress: bool,
) -> List[Dict[str, object]]:
    device = config['device']
    geo_dtype = config['geo_dtype']
    sim_dtype = config['sim_dtype']
    layer0_geometry = config['layer0_geometry']
    results: List[Dict[str, object]] = []
    monitor.mark('example2_wavelength_sweep_start', metadata={'count': len(wavelengths_nm), 'order': int(order_N)})

    progress_bar = None
    if progress and tqdm is not None:
        progress_bar = tqdm(wavelengths_nm, desc='Wavelength Sweep', unit='nm')
        wavelength_iter: Iterable[float] = progress_bar
    else:
        wavelength_iter = wavelengths_nm

    try:
        for lamb in wavelength_iter:
            meta = {'wavelength_nm': float(lamb), 'order': int(order_N)}
            with working_directory(EXAMPLE_DIR):
                lamb0 = torch.tensor(float(lamb), dtype=geo_dtype, device=device)
                silicon_eps = Materials.aSiH.apply(lamb0) ** 2
            layer0_eps = layer0_geometry * silicon_eps + (1.0 - layer0_geometry)
            sim = torcwa.rcwa(freq=1.0 / lamb0, order=[order_N, order_N], L=config['L'], dtype=sim_dtype, device=device)
            sim.add_input_layer(eps=config['substrate_eps'])
            sim.set_incident_angle(inc_ang=config['inc_ang'], azi_ang=config['azi_ang'])
            sim.add_layer(thickness=config['layer0_thickness'], eps=layer0_eps)

            monitor.mark('example2_wavelength_smatrix_start', metadata=meta)
            sim.solve_global_smatrix()
            monitor.mark('example2_wavelength_smatrix_end', metadata=meta, sync_cuda=True)

            txx = sim.S_parameters(orders=[0, 0], direction='forward', port='transmission', polarization='xx', ref_order=[0, 0])
            tyy = sim.S_parameters(orders=[0, 0], direction='forward', port='transmission', polarization='yy', ref_order=[0, 0])
            monitor.mark('example2_wavelength_sparams', metadata=meta, sync_cuda=True)

            results.append({
                'wavelength_nm': meta['wavelength_nm'],
                'order': int(order_N),
                'txx': complex(txx.detach().cpu().item()),
                'tyy': complex(tyy.detach().cpu().item()),
            })

            if progress_bar is not None:
                progress_bar.set_postfix(wavelength_nm=f"{meta['wavelength_nm']:.1f}")

            if store_transfer:
                out_path = output_dir / f"example2_lambda_{int(round(lamb))}_Sparams.mat"
                sio.savemat(str(out_path), {'txx': txx.cpu().numpy(), 'tyy': tyy.cpu().numpy(), 'wavelength_nm': float(lamb)})

            del sim
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if progress_bar is not None:
            progress_bar.close()

    monitor.mark('example2_wavelength_sweep_end', metadata={'count': len(wavelengths_nm)})
    return results


def _example6_objective(rho: torch.Tensor, cfg: Dict[str, object]) -> torch.Tensor:
    sim = torcwa.rcwa(freq=1.0 / cfg["lamb0"], order=[15, 8], L=cfg["L"], dtype=cfg["sim_dtype"], device=cfg["device"])
    sim.add_input_layer(eps=cfg["substrate_eps"])
    sim.set_incident_angle(inc_ang=cfg["inc_ang"], azi_ang=cfg["azi_ang"])
    layer0_eps = rho * cfg["silicon_eps"] + (1.0 - rho)
    sim.add_layer(thickness=cfg["layer0_thickness"], eps=layer0_eps)
    sim.solve_global_smatrix()
    t1xx = sim.S_parameters(orders=[1, 0], direction="forward", port="transmission", polarization="xx", ref_order=[0, 0])
    t1yy = sim.S_parameters(orders=[1, 0], direction="forward", port="transmission", polarization="yy", ref_order=[0, 0])
    t1xy = sim.S_parameters(orders=[1, 0], direction="forward", port="transmission", polarization="xy", ref_order=[0, 0])
    t1yx = sim.S_parameters(orders=[1, 0], direction="forward", port="transmission", polarization="yx", ref_order=[0, 0])
    return torch.abs(t1xx) ** 2 + torch.abs(t1yy) ** 2 + torch.abs(t1xy) ** 2 + torch.abs(t1yx) ** 2


def run_example6_optimization(
    monitor: ResourceMonitor,
    cfg: Dict[str, object],
    iterations: int,
    output_dir: Path,
    mark_every: int,
    progress: bool,
) -> Dict[str, List[float]]:
    geo_dtype = cfg['geo_dtype']
    device = cfg['device']
    Lx, Ly = cfg['L']
    dx = Lx / torcwa.rcwa_geo.nx
    dy = Ly / torcwa.rcwa_geo.ny
    x_kernel_axis = (torch.arange(torcwa.rcwa_geo.nx, dtype=geo_dtype, device=device) - (torcwa.rcwa_geo.nx - 1) / 2.0) * dx
    y_kernel_axis = (torch.arange(torcwa.rcwa_geo.ny, dtype=geo_dtype, device=device) - (torcwa.rcwa_geo.ny - 1) / 2.0) * dy
    x_kernel_grid, y_kernel_grid = torch.meshgrid(x_kernel_axis, y_kernel_axis, indexing='ij')
    blur_radius = 20.0
    g = torch.exp(-(x_kernel_grid ** 2 + y_kernel_grid ** 2) / blur_radius ** 2)
    g = g / torch.sum(g)
    g_fft = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(g)))

    torch.manual_seed(333)
    rho = torch.rand((torcwa.rcwa_geo.nx, torcwa.rcwa_geo.ny), dtype=geo_dtype, device=device)
    rho = (rho + torch.fliplr(rho)) / 2.0
    rho_fft = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(rho)))
    rho = torch.real(torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(rho_fft * g_fft))))
    momentum = torch.zeros_like(rho)
    velocity = torch.zeros_like(rho)

    gar_initial = 0.02
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1.0e-8
    iter_max = iterations
    beta = np.exp(np.arange(start=0, stop=iter_max) * np.log(1000.0) / iter_max)
    gar = gar_initial * 0.5 * (1 + np.cos(np.arange(start=0, stop=iter_max) * np.pi / iter_max))

    FoM_history: List[float] = []
    iteration_wall: List[float] = []

    start_time = time.time()
    monitor.mark('example6_optimization_start', metadata={'iterations': iterations})

    iter_indices = range(iter_max)
    progress_bar = None
    if progress and tqdm is not None:
        progress_bar = tqdm(iter_indices, desc='Optimization', unit='iter')
        iteration_iter: Iterable[int] = progress_bar
    else:
        iteration_iter = iter_indices

    for it in iteration_iter:
        iter_meta = {'iteration': int(it)}
        rho.requires_grad_(True)
        rho_fft = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(rho)))
        rho_bar = torch.real(torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(rho_fft * g_fft))))
        rho_tilda = 0.5 + torch.tanh(2 * beta[it] * rho_bar - beta[it]) / (2 * np.math.tanh(beta[it]))

        if it % mark_every == 0:
            monitor.mark('example6_objective_start', metadata=iter_meta)
        FoM = _example6_objective(rho_tilda, cfg)
        if it % mark_every == 0:
            monitor.mark('example6_objective_end', metadata=iter_meta, sync_cuda=True)

        if it % mark_every == 0:
            monitor.mark('example6_backward_start', metadata=iter_meta)
        FoM.backward()
        if it % mark_every == 0:
            monitor.mark('example6_backward_end', metadata=iter_meta, sync_cuda=True)

        with torch.no_grad():
            rho_gradient = rho.grad
            rho.grad = None

            FoM_value = float(FoM.detach().cpu().numpy())
            FoM_history.append(FoM_value)
            iteration_wall.append(time.time() - start_time)

            if progress_bar is not None:
                progress_bar.set_postfix(FoM=f'{FoM_value:.4f}')

            if it % mark_every == 0:
                monitor.mark('example6_update_start', metadata=iter_meta)
            momentum = (beta1 * momentum + (1 - beta1) * rho_gradient)
            velocity = (beta2 * velocity + (1 - beta2) * (rho_gradient ** 2))
            rho += gar[it] * (momentum / (1 - beta1 ** (it + 1))) / torch.sqrt((velocity / (1 - beta2 ** (it + 1))) + epsilon)
            rho = torch.clamp(rho, 0.0, 1.0)
            rho = (rho + torch.fliplr(rho)) / 2.0
            if it % mark_every == 0:
                monitor.mark('example6_update_end', metadata=iter_meta, sync_cuda=True)

        if (it + 1) % max(1, iter_max // 20) == 0:
            message = f'Iteration {it + 1}/{iter_max} FoM={FoM_value:.5f}'
            if progress_bar is not None:
                progress_bar.write(message)
            else:
                print(message)

    if progress_bar is not None:
        progress_bar.close()

    monitor.mark('example6_optimization_end', metadata={'iterations': iterations})

    np.savez(output_dir / 'example6_FoM_history.npz', FoM=np.array(FoM_history), wall_time=np.array(iteration_wall))
    return {'FoM': FoM_history, 'wall_time': iteration_wall}

def plot_resource_usage(df, events: List[Dict[str, object]], output_path: Path) -> None:
    if df.empty:
        return

    time_axis = df["elapsed_s"]
    fig, ax_pct = plt.subplots(figsize=(12, 6))

    percent_series = [
        ("cpu_percent", "CPU %", "#1f77b4"),
        ("process_cpu_percent", "Process CPU %", "#aec7e8"),
        ("gpu_util_percent", "GPU %", "#ff7f0e"),
        ("system_memory_percent", "System RAM %", "#2ca02c"),
        ("process_memory_percent", "Process RAM %", "#98df8a"),
        ("gpu_memory_percent", "GPU Mem %", "#d62728"),
    ]
    for column, label, color in percent_series:
        if column in df and df[column].notna().any():
            ax_pct.plot(time_axis, df[column], label=label, color=color)

    ax_pct.set_xlabel("Time [s]")
    ax_pct.set_ylabel("Utilization [%]")
    ax_pct.set_ylim(0, 110)
    ax_pct.grid(alpha=0.15)

    if len(time_axis) > 1:
        ax_pct.set_xlim(float(time_axis.min()), float(time_axis.max()))

    """
    ax_mb = ax_pct.twinx()
    mb_series = [
        ("system_memory_used_mb", "System RAM [MB]", "#2ca02c"),
        ("process_memory_mb", "Process RAM [MB]", "#98df8a"),
        ("gpu_memory_mb", "GPU Mem [MB]", "#d62728"),
        ("cuda_allocated_mb", "CUDA Alloc [MB]", "#ff9896"),
    ]
    for column, label, color in mb_series:
        if column in df and df[column].notna().any():
            ax_mb.plot(time_axis, df[column], label=label, color=color, linestyle="--", alpha=0.6)

    ax_mb.set_ylabel("Memory [MB]")
    """
    y_text = ax_pct.get_ylim()[1] - 4
    for event in events:
        ts = event.get("time_s")
        if ts is None:
            continue
        ax_pct.axvline(ts, color="#7f7f7f", alpha=0.25, linewidth=1.25)
        label = event.get("label") or ""
        if not label:
            continue
        ax_pct.text(
            ts,
            y_text,
            label,
            rotation=90,
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=9,
            color="#202020",
            alpha=0.9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"),
        )

    handles_pct, labels_pct = ax_pct.get_legend_handles_labels()
    #handles_mb, labels_mb = ax_mb.get_legend_handles_labels()
    #ax_pct.legend(handles_pct + handles_mb, labels_pct + labels_mb, loc="upper left", fontsize=9)
    ax_pct.legend(handles_pct, labels_pct, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

def parse_orders(order_str: Optional[str]) -> Sequence[int]:
    if not order_str:
        return list(range(0, 30, 2))
    parts = [int(x.strip()) for x in order_str.split(",") if x.strip()]
    if not parts:
        raise ValueError("At least one order must be specified")
    return parts


def parse_wavelengths(args) -> Sequence[float]:
    if args.wavelengths:
        return [float(x) for x in args.wavelengths.split(",")]
    return np.linspace(args.lambda_start, args.lambda_stop, args.lambda_count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TORCWA benchmarks with resource monitoring")
    parser.add_argument("--scenario", choices=["order", "wavelength", "optimization"], required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--sampling-interval", type=float, default=0.5, help="Monitor sampling interval in seconds")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/output"))
    parser.add_argument("--orders", type=str, help="Comma-separated list of diffraction orders for the order sweep")
    parser.add_argument("--field-order", type=int, help="Order index to capture field probe during the order sweep")
    parser.add_argument("--store-fields", action="store_true", help="Persist field probe data when capturing fields")
    parser.add_argument("--store-transfer", action="store_true", help="Persist S-parameters for each step")
    parser.add_argument("--lambda-start", type=float, default=450.0)
    parser.add_argument("--lambda-stop", type=float, default=650.0)
    parser.add_argument("--lambda-count", type=int, default=11)
    parser.add_argument("--wavelengths", type=str, help="Comma-separated list of wavelengths (nm) for the sweep")
    parser.add_argument("--wavelength-order", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=200, help="Number of iterations for the optimization benchmark")
    parser.add_argument("--mark-every", type=int, default=10, help="Mark optimization internals every N iterations")
    parser.add_argument("--no-warmup", action="store_true", help="Skip RCWA warmup call")
    parser.add_argument("--plot-format", type=str, default="png")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    args = parser.parse_args()

    device = resolve_device(args.device)
    geo_dtype = torch.float32
    sim_dtype = torch.complex64

    if not args.no_warmup:
        rcwa_warmup(device, geo_dtype, sim_dtype)

    monitor = ResourceMonitor(interval_s=args.sampling_interval)
    orders = parse_orders(args.orders)
    wavelengths = parse_wavelengths(args)

    progress_enabled = (tqdm is not None) and (not args.no_progress)
    if not progress_enabled and not args.no_progress and tqdm is None:
        print("tqdm not installed; install tqdm to view progress bars")

    output_dir = ensure_output_dir(args.output_dir / args.scenario)
    log_path = output_dir / "resource_log.csv"
    events_path = output_dir / "events.json"
    plot_path = output_dir / f"resource_plot.{args.plot_format}"

    scenario_payload: Dict[str, object] = {}

    monitor.start()
    try:
        if args.scenario == "order":
            cfg = setup_example2(device, geo_dtype, sim_dtype)
            payload = run_example2_order_sweep(
                monitor,
                cfg,
                orders,
                output_dir,
                store_transfer=args.store_transfer,
                field_probe_order=args.field_order,
                store_field=args.store_fields,
                progress=progress_enabled,
            )
            scenario_payload["order_results"] = payload
        elif args.scenario == "wavelength":
            cfg = setup_example2(device, geo_dtype, sim_dtype)
            payload = run_example2_wavelength_sweep(
                monitor,
                cfg,
                wavelengths,
                args.wavelength_order,
                output_dir,
                store_transfer=args.store_transfer,
                progress=progress_enabled,
            )
            scenario_payload["wavelength_results"] = payload
        elif args.scenario == "optimization":
            cfg = setup_example6(device, geo_dtype, sim_dtype)
            payload = run_example6_optimization(
                monitor,
                cfg,
                iterations=args.iterations,
                output_dir=output_dir,
                mark_every=max(1, args.mark_every),
                progress=progress_enabled,
            )
            scenario_payload["optimization_metrics"] = payload
    finally:
        monitor.stop()

    try:
        df = monitor.to_dataframe()
        ensure_output_dir(output_dir)
        df.to_csv(log_path, index=False)
        plot_resource_usage(df, monitor.events(), plot_path)
    except ImportError:
        print("pandas is required to export monitor data; please install pandas")
    with events_path.open("w", encoding="utf-8") as f:
        json.dump(monitor.events(), f, indent=2)

    meta = {
        "scenario": args.scenario,
        "device": str(device),
        "orders": list(map(int, orders)),
        "wavelengths": [float(x) for x in wavelengths],
        "iterations": args.iterations,
        "sampling_interval": args.sampling_interval,
        "progress_enabled": bool(progress_enabled),
        "timestamp": time.time(),
        "payload": scenario_payload,
    }
    with (output_dir / "benchmark_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)


if __name__ == "__main__":
    main()