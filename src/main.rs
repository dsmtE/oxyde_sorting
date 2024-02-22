mod app;

fn main() {
    oxyde::fern::Dispatch::new()
        .level(oxyde::log::LevelFilter::Debug)
        .level_for("wgpu_core", oxyde::log::LevelFilter::Warn)
        .level_for("wgpu_hal", oxyde::log::LevelFilter::Warn)
        .level_for("naga", oxyde::log::LevelFilter::Warn)
        .format(oxyde::default_formatter)
        // .chain(std::io::stdout())
        .chain(oxyde::fern::Dispatch::new().chain(std::io::stdout()).format(oxyde::color_formatter))
        .apply()
        .unwrap();

    oxyde::run_application::<app::App>(
        oxyde::AppConfig {
            is_resizable: true,
            title: "Sorting",
            control_flow: oxyde::winit::event_loop::ControlFlow::Poll,
        },
        oxyde::RenderingConfig {
            power_preference: oxyde::wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        },
    )
    .unwrap();
}
