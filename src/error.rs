#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("image width and height are not the same")]
    UnsquareImage,

    #[error("ort: {0}")]
    OnnxRuntime(#[from] ort::Error),

    #[error("incompatible onnx model")]
    IncompatibleModel,
    
    #[error("model io widths and heights are not the same")]
    UnsquareModelIO,

}
