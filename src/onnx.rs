use std::{fmt::Debug, path::Path};

use image::{DynamicImage, GenericImageView, RgbImage};
use ndarray::Array4;
use ort::{Session, ValueType};

use crate::{error::Error, upscaler::UpscaleSquareImage};

#[derive(Debug)]
pub struct ONNXNeuralUpscaler {
    session: Session,
    original_res: u32,
    target_res: u32,
    image: Array4<f32>,
    upscaled_image: DynamicImage,
    scale_factor: f32,
}

impl ONNXNeuralUpscaler {
    pub fn from_model(filepath: impl AsRef<Path>) -> Result<Self, Error> {
        fn validated_model_io_dims(iotype: &ValueType) -> Result<(u32, u32), Error> {
            let dims = match &iotype {
                ort::ValueType::Tensor { ty: _, dimensions } => dimensions.to_owned(),
                _ => return Err(Error::IncompatibleModel),
            };

            if dims[0] != 1 || dims[1] != 3 || dims.len() != 4 || dims[2] < 2 || dims[3] < 2 {
                return Err(Error::IncompatibleModel);
            }

            Ok((dims[2] as u32, dims[3] as u32))
        }

        let session = Session::builder()?.commit_from_file(filepath)?;
        let (x_in, y_in) = validated_model_io_dims(&session.inputs[0].input_type)?;
        let (x_out, y_out) = validated_model_io_dims(&session.outputs[0].output_type)?;

        if x_in != y_in || x_out != y_out {
            return Err(Error::UnsquareModelIO);
        }

        let upscaled_image = RgbImage::new(x_out, x_out);
        let scale_factor = x_out as f32 / x_in as f32;

        Ok(Self {
            session,
            original_res: x_in,
            target_res: x_out,
            image: Array4::zeros([1, 3, x_in as usize, x_in as usize]),
            upscaled_image: upscaled_image.into(),
            scale_factor,
        })
    }
}

impl UpscaleSquareImage for ONNXNeuralUpscaler {
    type Error = Error;

    fn load(&mut self, image: &DynamicImage) -> Result<(), Self::Error> {
        if image.width() != image.height() {
            return Err(Error::UnsquareImage);
        }

        let side = image.width();
        self.image = Array4::zeros([1, 3, side as usize, side as usize]);
        for (x, y, color) in image.pixels() {
            let (x, y) = (x as usize, y as usize);
            self.image[[0, 0, x, y]] = color[2] as f32;
            self.image[[0, 1, x, y]] = color[1] as f32;
            self.image[[0, 2, x, y]] = color[0] as f32;
        }
        Ok(())
    }

    fn upscale(&self) -> Result<DynamicImage, Self::Error> {
        let outputs = self
            .session
            .run(ort::inputs![&self.session.inputs[0].name => self.image.view()]?)?;

        let pixels: Vec<u8> = outputs[0]
            .try_extract_tensor::<f32>()?
            .iter()
            .map(|&i| i as u8)
            .collect();

        Ok(RgbImage::from_raw(
            self.upscaled_resolution(),
            self.upscaled_resolution(),
            pixels,
        )
        .unwrap()
        .into())
    }

    fn upscale_inplace(&mut self) -> Result<&DynamicImage, Self::Error> {
        self.upscaled_image = self.upscale()?;
        Ok(&self.upscaled_image)
    }

    fn upscale_factor(&self) -> f32 {
        self.scale_factor
    }

    fn original_resolution(&self) -> u32 {
        self.original_res
    }

    fn upscaled_resolution(&self) -> u32 {
        self.target_res
    }

    fn upscale_repeat(&mut self, times: usize) -> Result<&DynamicImage, Self::Error> {
        for _ in 0..times {
            self.upscale_inplace()?;
        }

        Ok(&self.upscaled_image)
    }
}
