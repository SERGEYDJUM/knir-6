#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("image width and height are not the same")]
    UnsquareImage,

    // #[error("ort: {0}")]
    // OnnxRuntime(#[from] ort::Error),

    #[error("incompatible onnx model")]
    IncompatibleModel,
    
    #[error("model io widths and heights are not the same")]
    UnsquareModelIO,

    #[error("wgpu: {0}")]
    FailedDeviceRequest(#[from] wgpu::RequestDeviceError),

    #[error("non-unicode symbols in path")]
    NonUnicodePath,

    #[error("io: {0}")]
    IO(#[from] std::io::Error),

    #[error("wgpu: {0}")]
    BufferFailedToMap(#[from] wgpu::BufferAsyncError),

    #[error("malformed final image")]
    MalformedOutput,
}
