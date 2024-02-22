use oxyde::{
    anyhow::Result,
    AppState,
    winit::event::Event,
    wgpu,
};

pub struct App {
}

impl oxyde::App for App {
    fn create(_app_state: &mut AppState) -> Self {
        Self {
        }
    }

    fn handle_event<T: 'static>(&mut self, _app_state: &mut AppState, _event: &Event<T>) -> Result<()> { Ok(()) }

    fn render_gui(&mut self, _app_state: &mut AppState) -> Result<()> { Ok(()) }

    fn update(&mut self, _app_state: &mut AppState) -> Result<()> { Ok(()) }

    fn render(&mut self, _app_state: &mut AppState, _output_view: &wgpu::TextureView) -> Result<()> { Ok(()) }

    fn post_render(&mut self, _app_state: &mut AppState) -> Result<()> { Ok(()) }
}
