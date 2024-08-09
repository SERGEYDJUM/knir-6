use criterion::{criterion_group, criterion_main, Criterion};
use image::imageops::FilterType;
use scale_benchmarks::{algo_scalers::CPUAlgoUpscaler, upscaler::UpscaleSquareImage};

fn criterion_benchmark(c: &mut Criterion) {
    let scaler = CPUAlgoUpscaler::new(2.0, FilterType::Lanczos3);
    c.bench_function("lanczos3", |b| b.iter(|| scaler.upscale().unwrap()));

    let scaler = CPUAlgoUpscaler::new(2.0, FilterType::CatmullRom);
    c.bench_function("catmullrom", |b| b.iter(|| scaler.upscale().unwrap()));

    let scaler = CPUAlgoUpscaler::new(2.0, FilterType::Gaussian);
    c.bench_function("gaussian", |b| b.iter(|| scaler.upscale().unwrap()));

    let scaler = CPUAlgoUpscaler::new(2.0, FilterType::Triangle);
    c.bench_function("triangle", |b| b.iter(|| scaler.upscale().unwrap()));

    let scaler = CPUAlgoUpscaler::new(2.0, FilterType::Nearest);
    c.bench_function("nearest", |b| b.iter(|| scaler.upscale().unwrap()));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
