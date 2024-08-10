use std::path::{Path, PathBuf};

use image::{DynamicImage, RgbImage};
use ndarray::Array4;
use ort::{Input, Session, ValueType};

use crate::{error::Error, upscaler::UpscaleSquareImage};

#[derive(Debug)]
pub struct ONNXNeuralUpscaler {
    session: Session,
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

        let image = RgbImage::new(x_in, x_in);
        let upscaled_image = RgbImage::new(x_out, x_out);
        let scale_factor = x_out as f32 / x_in as f32;

        Ok(Self {
            session,
            image: Array4::zeros([1, 3, x_in as usize, x_in as usize]),
            upscaled_image: upscaled_image.into(),
            scale_factor,
        })
    }
}

impl UpscaleSquareImage for ONNXNeuralUpscaler {
    type Error = Error;

    fn load(&mut self, image: &DynamicImage) -> Result<(), Self::Error> {
        todo!()
    }

    fn upscale(&self) -> Result<DynamicImage, Self::Error> {
        todo!()
    }

    fn upscale_inplace(&mut self) -> Result<&DynamicImage, Self::Error> {
        todo!()
    }

    fn upscale_factor(&self) -> f32 {
        todo!()
    }

    fn original_resolution(&self) -> u32 {
        todo!()
    }

    fn upscale_repeat(&mut self, times: usize) -> Result<&DynamicImage, Self::Error> {
        todo!()
    }
}