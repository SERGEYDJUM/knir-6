use criterion::{criterion_group, criterion_main, Criterion};
use image::imageops::FilterType;
use scale_benchmarks::{cpu_algo::CPUAlgoUpscaler, upscaler::UpscaleSquareImage};

fn cpu_algo(c: &mut Criterion) {
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

// fn cpu_nn(c: &mut Criterion) {
//     let scaler = ONNXNeuralUpscaler::from_model("models/realesr-general-wdn-x4v3.pth.onnx").unwrap();
//     scaler.upscale().unwrap();
    
//     c.bench_function("compact-x4", |b| b.iter(|| scaler.upscale().unwrap()));
// }

criterion_group!(benches, cpu_algo);
criterion_main!(benches);
