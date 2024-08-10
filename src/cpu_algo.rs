use image::{imageops::FilterType, DynamicImage, RgbImage};

use crate::{error::Error, upscaler::UpscaleSquareImage};

#[derive(Debug, Clone)]
pub struct CPUAlgoUpscaler {
    image: DynamicImage,
    upscaled_image: DynamicImage,
    scale_mode: FilterType,
    scale_factor: f32,
}

impl Default for CPUAlgoUpscaler {
    fn default() -> Self {
        let default_res = 512;
        let scale_factor = 2.0;

        let image = RgbImage::new(default_res, default_res);
        let new_res = (default_res as f32 * scale_factor) as u32;
        let upscaled_image = RgbImage::new(new_res, new_res);

        Self {
            image: image.into(),
            upscaled_image: upscaled_image.into(),
            scale_mode: FilterType::Nearest,
            scale_factor,
        }
    }
}

impl CPUAlgoUpscaler {
    pub fn new(scale_factor: f32, scale_mode: FilterType) -> Self {
        let mut scaler = Self {
            scale_mode,
            scale_factor,
            ..Default::default()
        };
        let upscaled_resolution = scaler.upscaled_resolution();
        scaler.upscaled_image = RgbImage::new(upscaled_resolution, upscaled_resolution).into();
        scaler
    }
}

impl UpscaleSquareImage for CPUAlgoUpscaler {
    type Error = Error;

    fn load(&mut self, image: &DynamicImage) -> Result<(), Self::Error> {
        if image.width() != image.height() {
            return Err(Error::UnsquareImage);
        }
        self.image = image.clone();
        Ok(())
    }

    fn upscale(&self) -> Result<DynamicImage, Self::Error> {
        Ok(self.image.resize(
            self.upscaled_resolution(),
            self.upscaled_resolution(),
            self.scale_mode,
        ))
    }

    fn upscale_inplace(&mut self) -> Result<&DynamicImage, Self::Error> {
        self.upscaled_image = self.upscale()?;
        Ok(&self.upscaled_image)
    }

    fn upscale_repeat(&mut self, times: usize) -> Result<&DynamicImage, Self::Error> {
        for _ in 0..times {
            self.upscale_inplace()?;
        }
        Ok(&self.upscaled_image)
    }

    fn upscale_factor(&self) -> f32 {
        self.scale_factor
    }

    fn original_resolution(&self) -> u32 {
        self.image.width()
    }
}
