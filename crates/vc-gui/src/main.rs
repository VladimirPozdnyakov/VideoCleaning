mod app;
mod export_ai;
mod masks_panel;
mod timeline;
mod tool_panel;
mod viewport;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_app_id("videocleaning")
            .with_inner_size([1280.0, 800.0])
            .with_title("VideoCleaning"),
        ..Default::default()
    };
    eframe::run_native(
        "VideoCleaning",
        options,
        Box::new(|_cc| Ok(Box::new(app::VcApp::new()))),
    )
}
