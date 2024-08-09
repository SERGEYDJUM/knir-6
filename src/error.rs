#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("image width and height are not the same")]
    UnsquareImage,
}
