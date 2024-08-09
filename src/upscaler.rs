use image::DynamicImage;

/// Allocation-effecient upscaling of square images
pub trait UpscaleSquareImage: Default {
    type Error;

    /// Stores an image in an optimal way, prepares it for upscaling
    fn load(&mut self, image: &DynamicImage) -> Result<(), Self::Error>;

    /// Upscales currently loaded image and returns it
    fn upscale(&self) -> Result<DynamicImage, Self::Error>;

    /// Upscales currently loaded image into Self's field
    fn upscale_inplace(&mut self) -> Result<&DynamicImage, Self::Error>;

    /// Returns the upscaling factor
    fn upscale_factor(&self) -> f32;

    /// Returns a resolution of the currently loaded image
    fn original_resolution(&self) -> u32;

    /// Returns a post-upscale resolution of the currently loaded image
    fn upscaled_resolution(&self) -> u32 {
        (self.original_resolution() as f32 * self.upscale_factor()) as u32
    }

    /// *Convenience function for benchmarking.*
    ///
    /// Repeats `upscale` multiple times with overwriting.
    fn upscale_repeat(&mut self, times: usize) -> Result<&DynamicImage, Self::Error>;
}
