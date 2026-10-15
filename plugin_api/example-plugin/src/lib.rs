//! A complete, small eov plugin.
//!
//! It shows the main parts of the plugin API:
//!
//! - a toolbar button that toggles a sidebar surface ([`SurfaceSpec`]),
//! - a point tool whose markers live in an overlay layer and can be undone,
//! - a context-menu item and a HUD button,
//! - a CPU viewport filter,
//! - persisted settings and a background job with progress.
//!
//! Build it with `cargo build --release -p eov-example-plugin` and package
//! `plugin.toml`, `ui/` and the library into a tar archive named
//! `example_plugin.eop` (see the README).

use eov_plugin_api::prelude::*;
use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

const SIDEBAR_BUTTON: &str = "toggle_sidebar";
const INVERT_BUTTON: &str = "toggle_invert";
const POINT_TOOL: &str = "place_marker";
const FILTER: &str = "invert";
/// Layer ids are unique per plugin across slides, hence one id per file.
fn layer_id(file_id: FileId) -> String {
    format!("markers-{file_id}")
}

const ICON: &str = "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><circle cx='12' cy='12' r='8' fill='none' stroke='white' stroke-width='2'/></svg>";

#[derive(Default)]
struct State {
    sidebar: Option<Surface>,
    /// Markers per open file, by object id.
    markers: BTreeMap<FileId, BTreeMap<String, Shape>>,
    /// Undo entries: entry id → (file, object id, shape).
    history: BTreeMap<String, (FileId, String, Shape)>,
    active_file: Option<FileInfo>,
}

pub struct ExamplePlugin {
    host: Host,
    state: Mutex<State>,
    invert: AtomicBool,
    next_id: AtomicU64,
}

impl ExamplePlugin {
    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        // A panic elsewhere must not wedge the plugin forever.
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn refresh_sidebar(&self, state: &State) {
        let Some(sidebar) = &state.sidebar else {
            return;
        };
        let count = state
            .active_file
            .as_ref()
            .and_then(|file| state.markers.get(&file.file_id))
            .map_or(0, BTreeMap::len);
        let mut properties = Map::new();
        properties.insert("marker-count".into(), json!(count));
        properties.insert(
            "active-file".into(),
            json!(
                state
                    .active_file
                    .as_ref()
                    .map_or("No slide".to_string(), |f| f.filename.clone())
            ),
        );
        properties.insert(
            "invert-enabled".into(),
            json!(self.invert.load(Ordering::Relaxed)),
        );
        let _ = sidebar.set_many(properties);
    }

    fn toggle_sidebar(&self) -> Result<(), String> {
        let mut state = self.state();
        if let Some(sidebar) = state.sidebar.take()
            && sidebar.is_open()
        {
            sidebar.close().map_err(String::from)?;
            self.host
                .set_toolbar_button_active(SIDEBAR_BUTTON, false)
                .map_err(String::from)?;
            return Ok(());
        }
        let surface = self
            .host
            .open_surface(
                SurfaceSpec::new(
                    SurfaceKind::Sidebar,
                    "ui/example-sidebar.slint",
                    "ExampleSidebar",
                )
                .with_size(260.0, None),
            )
            .map_err(String::from)?;
        state.sidebar = Some(surface);
        self.refresh_sidebar(&state);
        self.host
            .set_toolbar_button_active(SIDEBAR_BUTTON, true)
            .map_err(String::from)
    }

    fn set_invert(&self, enabled: bool) -> Result<(), String> {
        self.invert.store(enabled, Ordering::Relaxed);
        self.host
            .set_filter_enabled(FILTER, enabled)
            .map_err(String::from)?;
        let _ = self.host.update_hud_button(ButtonUpdate {
            button_id: INVERT_BUTTON.into(),
            active: Some(enabled),
            ..Default::default()
        });
        self.host
            .set_setting("invert", &enabled)
            .map_err(String::from)?;
        self.refresh_sidebar(&self.state());
        Ok(())
    }

    fn ensure_layer(&self, file_id: FileId) -> Result<Layer, String> {
        self.host
            .create_layer(
                LayerSpec::new(layer_id(file_id), file_id)
                    .named("Markers")
                    .interactive(true)
                    .editable(true)
                    .with_style(Style {
                        stroke: Rgba::rgb(0xFF, 0x9F, 0x1C),
                        point_radius: 6.0,
                        ..Style::default()
                    }),
            )
            .map_err(String::from)
    }

    fn redraw_layer(&self, state: &State, file_id: FileId) -> Result<(), String> {
        let objects = state
            .markers
            .get(&file_id)
            .into_iter()
            .flatten()
            .map(|(id, shape)| {
                OverlayObject::new(id.clone(), shape.clone()).with_tooltip(format!("Marker {id}"))
            })
            .collect();
        self.host
            .layer(layer_id(file_id))
            .replace_all(objects)
            .map_err(String::from)
    }

    fn add_marker(&self, file_id: FileId, x: f64, y: f64) -> Result<(), String> {
        let id = format!("m{}", self.next_id.fetch_add(1, Ordering::Relaxed));
        let shape = Shape::Point { x, y };
        let mut state = self.state();
        state
            .markers
            .entry(file_id)
            .or_default()
            .insert(id.clone(), shape.clone());
        let entry_id = format!("add-{id}");
        state.history.insert(entry_id.clone(), (file_id, id, shape));
        self.redraw_layer(&state, file_id)?;
        self.refresh_sidebar(&state);
        self.host
            .push_undo(entry_id, Some(file_id), "Add marker")
            .map_err(String::from)
    }

    /// Undoes (`present = false`) or redoes an "add marker" entry.
    fn apply_history(&self, entry_id: &str, present: bool) -> Result<(), String> {
        let mut state = self.state();
        let Some((file_id, id, shape)) = state.history.get(entry_id).cloned() else {
            return Ok(());
        };
        let markers = state.markers.entry(file_id).or_default();
        if present {
            markers.insert(id, shape);
        } else {
            markers.remove(&id);
        }
        self.redraw_layer(&state, file_id)?;
        self.refresh_sidebar(&state);
        Ok(())
    }

    fn run_job(&self) -> Result<(), String> {
        let job = self
            .host
            .start_job("Counting to ten", true)
            .map_err(String::from)?;
        std::thread::spawn(move || {
            for step in 1..=10 {
                std::thread::sleep(Duration::from_millis(100));
                job.update(Some(step as f32 / 10.0), Some(&format!("{step} of 10")));
            }
            job.finish(None);
        });
        Ok(())
    }
}

impl Plugin for ExamplePlugin {
    fn create(host: Host, init: PluginInit) -> Result<Self, String> {
        host.add_toolbar_button(ToolbarButtonSpec {
            button_id: SIDEBAR_BUTTON.into(),
            tooltip: "Example sidebar".into(),
            icon_svg: ICON.into(),
            ..Default::default()
        })
        .map_err(String::from)?;
        host.register_tool(
            ToolSpec::new(POINT_TOOL, "Place marker", ICON, ToolKind::Point)
                .with_hotkey(Hotkey::key("k")),
        )
        .map_err(String::from)?;
        host.add_hud_button(HudButtonSpec {
            button_id: INVERT_BUTTON.into(),
            tooltip: "Invert colours".into(),
            icon_svg: ICON.into(),
            ..Default::default()
        })
        .map_err(String::from)?;
        let invert = init
            .settings
            .get("invert")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        host.register_filter(FilterSpec {
            filter_id: FILTER.into(),
            name: "Invert".into(),
            priority: 0,
            enabled: invert,
        })
        .map_err(String::from)?;
        Ok(Self {
            host,
            state: Mutex::new(State::default()),
            invert: AtomicBool::new(invert),
            next_id: AtomicU64::new(1),
        })
    }

    fn on_event(&self, event: HostEvent) -> Result<(), String> {
        match event {
            HostEvent::Ready { restore_sidebar } => {
                if restore_sidebar {
                    self.toggle_sidebar()?;
                }
            }
            HostEvent::FileOpened { file } => {
                self.ensure_layer(file.file_id)?;
                let state = self.state();
                self.redraw_layer(&state, file.file_id)?;
            }
            HostEvent::FileClosed { file_id } => {
                let mut state = self.state();
                state.markers.remove(&file_id);
                state.history.retain(|_, (file, _, _)| *file != file_id);
            }
            HostEvent::ActiveChanged { file_id, .. } => {
                let snapshot = self.host.snapshot().map_err(String::from)?;
                let mut state = self.state();
                state.active_file = file_id.and_then(|id| snapshot.file(id).cloned());
                self.refresh_sidebar(&state);
            }
            HostEvent::ToolbarButtonClicked { button_id } if button_id == SIDEBAR_BUTTON => {
                self.toggle_sidebar()?;
            }
            HostEvent::HudButtonClicked { button_id, .. } if button_id == INVERT_BUTTON => {
                let enabled = !self.invert.load(Ordering::Relaxed);
                self.set_invert(enabled)?;
            }
            HostEvent::FilterToggled { filter_id, enabled } if filter_id == FILTER => {
                self.set_invert(enabled)?;
            }
            HostEvent::ShapeDrawn {
                tool_id,
                file_id,
                shape: Shape::Point { x, y },
                ..
            } if tool_id == POINT_TOOL => {
                self.add_marker(file_id, x, y)?;
            }
            HostEvent::ContextMenuItemSelected { item_id, context } if item_id == "add_marker" => {
                self.add_marker(context.view.file_id, context.image_x, context.image_y)?;
            }
            HostEvent::ObjectEdited { object, shape } => {
                let mut state = self.state();
                for (file_id, markers) in state.markers.iter_mut() {
                    if object.layer_id == layer_id(*file_id) {
                        markers.insert(object.object_id.clone(), shape.clone());
                    }
                }
            }
            HostEvent::ObjectsDeleteRequested { objects } => {
                let mut state = self.state();
                let Some(file_id) = state.active_file.as_ref().map(|f| f.file_id) else {
                    return Ok(());
                };
                if let Some(markers) = state.markers.get_mut(&file_id) {
                    for object in objects.iter().filter(|o| o.layer_id == layer_id(file_id)) {
                        markers.remove(&object.object_id);
                    }
                }
                self.redraw_layer(&state, file_id)?;
                self.refresh_sidebar(&state);
            }
            HostEvent::Undo { entry_id } => self.apply_history(&entry_id, false)?,
            HostEvent::Redo { entry_id } => self.apply_history(&entry_id, true)?,
            HostEvent::SurfaceClosed { surface_id } => {
                let mut state = self.state();
                if state.sidebar.as_ref().is_some_and(|s| s.id() == surface_id) {
                    state.sidebar = None;
                    let _ = self.host.set_toolbar_button_active(SIDEBAR_BUTTON, false);
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn context_menu(&self, _context: &MenuContext) -> Vec<MenuItem> {
        vec![MenuItem {
            item_id: "add_marker".into(),
            label: "Add marker here".into(),
            ..Default::default()
        }]
    }

    fn ui_callback(&self, _surface_id: &str, name: &str, args: &[Value]) -> Result<Value, String> {
        match name {
            "clear-markers" => {
                let mut state = self.state();
                if let Some(file_id) = state.active_file.as_ref().map(|f| f.file_id) {
                    state.markers.remove(&file_id);
                    self.redraw_layer(&state, file_id)?;
                    self.host.clear_undo(Some(file_id)).map_err(String::from)?;
                }
                self.refresh_sidebar(&state);
                Ok(Value::Null)
            }
            "toggle-invert" => {
                let enabled = args.first().and_then(Value::as_bool).unwrap_or(false);
                self.set_invert(enabled)?;
                Ok(Value::Null)
            }
            "start-job" => {
                self.run_job()?;
                Ok(Value::Null)
            }
            "describe" => {
                let state = self.state();
                let total: usize = state.markers.values().map(BTreeMap::len).sum();
                Ok(json!(format!(
                    "{total} marker(s) across {} slide(s)",
                    state.markers.len()
                )))
            }
            _ => Ok(Value::Null),
        }
    }

    fn apply_filter(&self, filter_id: &str, frame: &mut Frame<'_>) -> Result<(), String> {
        if filter_id == FILTER {
            for row in frame.rows_mut() {
                for px in row.as_chunks_mut::<4>().0 {
                    px[0] = 255 - px[0];
                    px[1] = 255 - px[1];
                    px[2] = 255 - px[2];
                }
            }
        }
        Ok(())
    }
}

eov_plugin_api::export_plugin!(ExamplePlugin);

/// The plugin's root module, for hosts that link it in-process (tests).
pub fn module() -> eov_plugin_api::abi::PluginModuleRef {
    eov_plugin_api::sdk::root_module::<ExamplePlugin>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use eov_plugin_api::testing::{MockHost, TestHarness, test_init};

    fn harness() -> TestHarness<ExamplePlugin> {
        let host = MockHost::new();
        host.state().snapshot.files.push(FileInfo {
            file_id: 1,
            filename: "slide.svs".into(),
            ..Default::default()
        });
        TestHarness::with_host(host, test_init("example_plugin")).unwrap()
    }

    #[test]
    fn registers_its_contributions() {
        let h = harness();
        let state = h.state();
        assert!(state.toolbar_buttons.contains_key(SIDEBAR_BUTTON));
        assert!(state.tools.contains_key(POINT_TOOL));
        assert!(state.hud_buttons.contains_key(INVERT_BUTTON));
        assert!(!state.filters[FILTER].enabled);
    }

    #[test]
    fn markers_are_drawn_and_undone() {
        let h = harness();
        h.send(HostEvent::FileOpened {
            file: FileInfo {
                file_id: 1,
                ..Default::default()
            },
        })
        .unwrap();
        h.send(HostEvent::ActiveChanged {
            pane: Some(1),
            file_id: Some(1),
        })
        .unwrap();
        h.send(HostEvent::ShapeDrawn {
            tool_id: POINT_TOOL.into(),
            pane: 1,
            file_id: 1,
            shape: Shape::Point { x: 5.0, y: 6.0 },
            modifiers: Modifiers::default(),
        })
        .unwrap();
        let layer = layer_id(1);
        assert_eq!(h.state().objects(&layer).len(), 1);
        let entry = h.state().undo[0].entry_id.clone();
        h.send(HostEvent::Undo {
            entry_id: entry.clone(),
        })
        .unwrap();
        assert!(h.state().objects(&layer).is_empty());
        h.send(HostEvent::Redo { entry_id: entry }).unwrap();
        assert_eq!(h.state().objects(&layer).len(), 1);
    }

    #[test]
    fn the_sidebar_round_trips_through_callbacks() {
        let h = harness();
        h.send(HostEvent::ToolbarButtonClicked {
            button_id: SIDEBAR_BUTTON.into(),
        })
        .unwrap();
        let surface_id = h.state().surfaces.keys().next().unwrap().clone();
        assert!(h.state().surfaces[&surface_id].open);
        let reply = h.ui_callback(&surface_id, "describe", vec![]).unwrap();
        assert_eq!(reply, json!("0 marker(s) across 0 slide(s)"));
        h.ui_callback(&surface_id, "toggle-invert", vec![json!(true)])
            .unwrap();
        assert!(h.state().filters[FILTER].enabled);
        assert_eq!(h.state().settings.get("invert"), Some(&json!(true)));
    }

    #[test]
    fn the_filter_inverts_rgb() {
        let h = harness();
        let mut pixels = vec![0, 10, 255, 255];
        h.apply_filter(FILTER, &mut pixels, 1, 1, FilterView::default())
            .unwrap();
        assert_eq!(pixels, vec![255, 245, 0, 255]);
    }
}
