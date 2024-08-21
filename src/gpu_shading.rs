use std::{borrow::Cow, fs::File, io::Read, path::Path, sync::mpsc};

use crate::{error::Error, upscaler::UpscaleSquareImage};
use image::{DynamicImage, RgbImage, RgbaImage};
use pollster::FutureExt;
use wgpu::{
    PipelineLayoutDescriptor, ShaderModuleDescriptor,
    TextureDescriptor, TextureUsages,
};

#[derive(Debug)]
struct GPUShadingUpscaler {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    output: OutputTex,
}

#[derive(Debug)]
struct OutputTex {
    size: wgpu::Extent3d,
    buffer_handle: wgpu::Buffer,
    texture_handle: wgpu::Texture,
}

impl GPUShadingUpscaler {
    fn from_image(
        shader_path: impl AsRef<Path>,
        image: &DynamicImage,
        scale_factor: f32,
    ) -> Result<Self, Error> {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .block_on()
            .unwrap();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .block_on()?;

        let vertex_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("GPUSU_ShaderModuleDescriptor_Vertex"),
            source: wgpu::ShaderSource::Wgsl(Cow::from(include_str!("vertex_plane.wgsl"))),
        });

        let fragment_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("GPUSU_ShaderModuleDescriptor_Fragment"),
            source: wgpu::ShaderSource::Wgsl({
                let mut shader_code = String::new();
                File::open(shader_path)?.read_to_string(&mut shader_code)?;
                Cow::from(shader_code)
            }),
        });

        let in_texture_size = wgpu::Extent3d {
            width: image.width(),
            height: image.height(),
            depth_or_array_layers: 1,
        };

        let in_texture_handle = device.create_texture(&TextureDescriptor {
            label: Some("GPUSU_InputTextureHandle"),
            size: in_texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
            label: Some("GPUSU_BindGroupLayout"),
        });

        let out_texture_size = wgpu::Extent3d {
            width: (image.width() as f32 * scale_factor) as u32,
            height: (image.height() as f32 * scale_factor) as u32,
            depth_or_array_layers: 1,
        };

        let out_texture_handle = device.create_texture(&TextureDescriptor {
            label: Some("GPUSU_OutputTextureHandle"),
            size: out_texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::COPY_SRC | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
        });

        let out_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPUSU_OutputBuffer"),
            size: out_texture_size.width as u64 * out_texture_size.height as u64 * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &in_texture_handle.create_view(&wgpu::TextureViewDescriptor::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&device.create_sampler(
                        &wgpu::SamplerDescriptor {
                            address_mode_u: wgpu::AddressMode::ClampToEdge,
                            address_mode_v: wgpu::AddressMode::ClampToEdge,
                            address_mode_w: wgpu::AddressMode::ClampToEdge,
                            mag_filter: wgpu::FilterMode::Linear,
                            min_filter: wgpu::FilterMode::Nearest,
                            mipmap_filter: wgpu::FilterMode::Nearest,
                            ..Default::default()
                        },
                    )),
                },
            ],
            label: Some("GPUSU_BindGroup"),
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("GPUSU_PipelineLayout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("GPUSU_Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &vertex_shader,
                    entry_point: "vs_main",
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &fragment_shader,
                    entry_point: "main",
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::TextureFormat::Rgba8UnormSrgb.into())],
                }),
                multiview: None,
                cache: None,
            },
        );

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &in_texture_handle,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &image.to_rgba8(),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * image.width()),
                rows_per_image: Some(image.height()),
            },
            in_texture_size,
        );

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group,
            output: OutputTex {
                size: out_texture_size,
                buffer_handle: out_staging_buffer,
                texture_handle: out_texture_handle,
            },
        })
    }

    fn queue_render(&mut self) {
        let mut command_encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("GPUSU_RenderPass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.output.texture_handle.create_view(&wgpu::TextureViewDescriptor::default()),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }

        command_encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.output.texture_handle,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &self.output.buffer_handle,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(self.output.size.width * 4),
                    rows_per_image: Some(self.output.size.height),
                },
            },
            self.output.size,
        );

        self.queue.submit(Some(command_encoder.finish()));
    }

    fn get_rendered_image(&self) -> Result<RgbaImage, Error> {
        let (sender, receiver) = mpsc::channel();

        let buffer_slice = self.output.buffer_handle.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, move |r| sender.send(r).unwrap());
        self.device.poll(wgpu::Maintain::wait()).panic_on_timeout();
        receiver.recv().unwrap()?;

        let output_raw = {
            let mut cpu_buffer = Vec::with_capacity(self.output.size.height as usize * self.output.size.width as usize * 4);
            let view = buffer_slice.get_mapped_range();
            cpu_buffer.extend_from_slice(&view[..]);
            cpu_buffer
        };

        match RgbaImage::from_raw(self.output.size.width, self.output.size.height, output_raw) {
            Some(image) => Ok(image),
            None => Err(Error::MalformedOutput),
        }
    }

    fn new(path: impl AsRef<Path>, scale_factor: f32) -> Result<Self, Error> {
        Self::from_image(path, &RgbImage::new(512, 512).into(), scale_factor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_test() {
        let image = image::open("target/input.jpeg").unwrap().crop(0, 128, 1024, 1024);
        let mut scaler = GPUShadingUpscaler::from_image("shaders/passthrough.wgsl", &image, 2.0).unwrap();

        scaler.queue_render();
        scaler.get_rendered_image().unwrap().save("target/test.png").unwrap();
    }
}
