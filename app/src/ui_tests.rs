//! The main window as assistive technology sees it: every control has a role
//! and a name, toggles report their state, and the accessible actions do what
//! a click does (AUDIT §14.4).
//!
//! The elements are found through `i-slint-backend-testing`'s
//! `ElementHandle`, which needs the element information that `build.rs`
//! emits in debug builds.

use crate::{
    AppWindow, ChannelRow, ChoiceItem, ExportFormat, HeatMapPanel, HudSettings,
    HudToolbarButtonData, JobData, MeasurementRow, NotificationData, PaneViewData, PluginRowData,
    PropertyRow, TabData,
};
use i_slint_backend_testing::{AccessibleRole, ElementHandle, ElementQuery};
use slint::{ComponentHandle, Model, ModelRc, VecModel};
use std::cell::Cell;
use std::rc::Rc;

/// Shows or hides one of the window's dialogs and panels.
type SetVisible = fn(&AppWindow, bool);

/// Roles of elements people operate: each needs a name.
const OPERATED: [AccessibleRole; 10] = [
    AccessibleRole::Button,
    AccessibleRole::Checkbox,
    AccessibleRole::Combobox,
    AccessibleRole::Slider,
    AccessibleRole::Spinbox,
    AccessibleRole::Tab,
    AccessibleRole::Switch,
    AccessibleRole::ListItem,
    AccessibleRole::RadioButton,
    AccessibleRole::TextInput,
];

/// A window tall enough to show the export dialog's settings unscrolled
/// (what is scrolled out of sight is hidden from assistive technology).
fn window() -> AppWindow {
    let ui = AppWindow::new().unwrap();
    ui.window().set_size(slint::PhysicalSize::new(1280, 2400));
    ui.show().unwrap();
    assert!(
        ElementHandle::find_by_element_type_name(&ui, "WindowToolbar")
            .next()
            .is_some(),
        "the UI tests need the element information of a debug build"
    );
    ui
}

/// The element on screen with this role and name.
pub(crate) fn find(ui: &AppWindow, role: AccessibleRole, label: &str) -> ElementHandle {
    ElementHandle::find_by_accessible_label(ui, label)
        .find(|element| element.accessible_role() == Some(role))
        .unwrap_or_else(|| panic!("no {role:?} is labelled {label:?}"))
}

/// The operated elements on screen that have no name.
fn unnamed(ui: &AppWindow) -> Vec<String> {
    ElementQuery::from_root(ui)
        .match_predicate(|element| {
            element
                .accessible_role()
                .is_some_and(|role| OPERATED.contains(&role))
                && element
                    .accessible_label()
                    .is_none_or(|label| label.is_empty())
        })
        .find_all()
        .into_iter()
        .map(|element| {
            format!(
                "{:?} {} ({:?})",
                element.accessible_role(),
                element.id().unwrap_or_default(),
                element.type_name().unwrap_or_default()
            )
        })
        .collect()
}

#[test]
fn every_control_has_a_name() {
    crate::test_support::run_on_slint_ui_test_thread(|| {
        let ui = window();
        assert!(operated(&ui) > 20, "the toolbar's controls are found");
        assert_eq!(unnamed(&ui), Vec::<String>::new(), "main window");
        fill(&ui);
        assert_eq!(unnamed(&ui), Vec::<String>::new(), "a slide with its HUD");
        // The dialogs and panels, one at a time.
        let surfaces: [(&str, SetVisible); 13] = [
            ("channels", AppWindow::set_channels_panel_visible),
            ("notifications", AppWindow::set_notifications_visible),
            ("export", AppWindow::set_export_dialog_visible),
            ("dataset", AppWindow::set_dataset_export_dialog_visible),
            ("slide info", AppWindow::set_slide_info_visible),
            ("save copy", AppWindow::set_save_copy_dialog_visible),
            ("plugins", AppWindow::set_plugins_panel_visible),
            ("bookmark", AppWindow::set_bookmark_dialog_visible),
            ("shortcuts", AppWindow::set_shortcuts_panel_visible),
            ("measurements", AppWindow::set_measurements_panel_visible),
            ("quantify", AppWindow::set_quantify_visible),
            ("lanczos", AppWindow::set_lanczos_confirm_visible),
            ("plugin confirm", AppWindow::set_plugin_confirm_visible),
        ];
        for (name, set_visible) in surfaces {
            set_visible(&ui, true);
            assert_eq!(unnamed(&ui), Vec::<String>::new(), "{name}");
            set_visible(&ui, false);
        }
        ui.global::<HeatMapPanel>().set_visible(true);
        assert_eq!(unnamed(&ui), Vec::<String>::new(), "heat map");
        ui.hide().unwrap();
    });
}

fn operated(ui: &AppWindow) -> usize {
    ElementQuery::from_root(ui)
        .match_predicate(|element| {
            element
                .accessible_role()
                .is_some_and(|role| OPERATED.contains(&role))
        })
        .find_all()
        .len()
}

fn model<T: Clone + 'static>(rows: Vec<T>) -> ModelRc<T> {
    ModelRc::new(VecModel::from(rows))
}

/// One pane with a slide, its HUD and the adjustments open, and rows in the
/// lists of the panels.
fn fill(ui: &AppWindow) {
    ui.set_panes(model(vec![PaneViewData {
        tabs: model(vec![TabData {
            id: 1,
            title: "slide.svs".into(),
            path: "/slides/slide.svs".into(),
            is_active: true,
            is_focused: true,
            ..Default::default()
        }]),
        zoom_display_text: "10×".into(),
        zoom_input_text: "10".into(),
        zoom_tooltip: "10× objective".into(),
        hud: HudSettings {
            show_hud_toolbar: true,
            hud_dropdown_open: true,
            gamma: 1.0,
            contrast: 1.0,
            scale_bar_width: 120.0,
            scale_bar_label: "100 µm".into(),
            ..Default::default()
        },
        hud_buttons: model(vec![HudToolbarButtonData {
            plugin_id: "p".into(),
            button_id: "b".into(),
            tooltip: "Plugin tool".into(),
            ..Default::default()
        }]),
        ..Default::default()
    }]));
    ui.set_channel_rows(model(vec![ChannelRow {
        index: 0,
        name: "DAPI".into(),
        min_text: "0".into(),
        max_text: "4095".into(),
        gamma_text: "1.00".into(),
        high: 1.0,
        gamma_position: 0.5,
        visible: true,
        ..Default::default()
    }]));
    ui.set_channel_chip_colors(model(vec![slint::Color::from_rgb_u8(255, 0, 0)]));
    ui.set_plugin_rows(model(vec![PluginRowData {
        id: "annotations".into(),
        name: "Annotations".into(),
        version: "0.2.4".into(),
        status: "Active".into(),
        update: "0.2.5".into(),
        can_disable: true,
        can_remove: true,
        ..Default::default()
    }]));
    ui.set_measurement_rows(model(vec![MeasurementRow {
        id: "m1".into(),
        kind: "Distance".into(),
        value: "12.5 µm".into(),
    }]));
    ui.set_slide_info_rows(model(vec![PropertyRow {
        key: "openslide.vendor".into(),
        value: "aperio".into(),
    }]));
    ui.set_notifications(model(vec![NotificationData {
        level: 3,
        message: "Something failed".into(),
        ..Default::default()
    }]));
    ui.set_jobs(model(vec![JobData {
        id: "j".into(),
        title: "Exporting".into(),
        fraction: 0.5,
        cancellable: true,
        ..Default::default()
    }]));
    let mut dataset = ui.get_dataset_export_settings();
    dataset.inputs = model(vec!["/slides/a.svs".into()]);
    ui.set_dataset_export_settings(dataset);
    ui.set_quantify_presets(model(vec![ChoiceItem {
        id: "hdab".into(),
        label: "H-DAB".into(),
    }]));
}

#[test]
fn slide_view_controls_report_their_state() {
    crate::test_support::run_on_slint_ui_test_thread(|| {
        let ui = window();
        fill(&ui);
        let tab = find(&ui, AccessibleRole::Tab, "slide.svs");
        assert_eq!(tab.accessible_item_selected(), Some(true));
        assert_eq!(
            tab.accessible_description().as_deref(),
            Some("/slides/slide.svs")
        );
        let closed = Rc::new(Cell::new(-1));
        ui.on_tab_close_requested({
            let closed = Rc::clone(&closed);
            move |_, id| closed.set(id)
        });
        find(&ui, AccessibleRole::Button, "Close slide.svs").invoke_accessible_default_action();
        assert_eq!(closed.get(), 1);

        let zoom = find(&ui, AccessibleRole::Spinbox, "Zoom");
        assert_eq!(zoom.accessible_value().as_deref(), Some("10×"));
        let zoomed_in = Rc::new(Cell::new(false));
        ui.on_hud_zoom_in({
            let zoomed_in = Rc::clone(&zoomed_in);
            move |_| zoomed_in.set(true)
        });
        zoom.invoke_accessible_increment_action();
        assert!(zoomed_in.get());

        let adjustments = find(&ui, AccessibleRole::Button, "Adjustments");
        assert_eq!(adjustments.accessible_expandable(), Some(true));
        assert_eq!(adjustments.accessible_expanded(), Some(true));
        let gamma = Rc::new(Cell::new(0.0));
        ui.on_hud_gamma_changed({
            let gamma = Rc::clone(&gamma);
            move |_, value| gamma.set(value)
        });
        find(&ui, AccessibleRole::Slider, "Gamma").invoke_accessible_increment_action();
        assert!((gamma.get() - 1.05).abs() < 1e-4, "{}", gamma.get());
        let unit = find(&ui, AccessibleRole::RadioButton, "Millimetres (mm)");
        assert_eq!(unit.accessible_checked(), Some(false));
        assert_eq!(
            find(&ui, AccessibleRole::RadioButton, "Micrometres (µm)").accessible_checked(),
            Some(true)
        );

        assert_eq!(
            find(&ui, AccessibleRole::Image, "Minimap")
                .accessible_description()
                .as_deref(),
            Some("The whole slide; click or drag to move the view there")
        );
        // One name for the scale bar, not one per outline copy.
        assert_eq!(
            ElementHandle::find_by_accessible_label(&ui, "Scale bar: 100 µm").count(),
            1
        );
        ui.hide().unwrap();
    });
}

#[test]
fn toolbar_toggles_report_their_state_and_act_on_their_action() {
    crate::test_support::run_on_slint_ui_test_thread(|| {
        let ui = window();
        let toggled = Rc::new(Cell::new(0));
        ui.on_toggle_minimap_requested({
            let toggled = Rc::clone(&toggled);
            move || toggled.set(toggled.get() + 1)
        });
        let minimap = find(&ui, AccessibleRole::Button, "Toggle minimap");
        assert_eq!(minimap.accessible_role(), Some(AccessibleRole::Button));
        assert_eq!(minimap.accessible_checkable(), Some(true));
        assert_eq!(minimap.accessible_checked(), Some(true));
        minimap.invoke_accessible_default_action();
        assert_eq!(toggled.get(), 1);
        ui.set_show_minimap(false);
        assert_eq!(minimap.accessible_checked(), Some(false));

        // Undo is a plain button, off until there is something to undo.
        let undo = find(&ui, AccessibleRole::Button, "Undo");
        assert_eq!(undo.accessible_checkable(), Some(false));
        assert_eq!(undo.accessible_enabled(), Some(false));
        let undone = Rc::new(Cell::new(false));
        ui.on_undo_requested({
            let undone = Rc::clone(&undone);
            move || undone.set(true)
        });
        undo.invoke_accessible_default_action();
        assert!(!undone.get(), "a disabled button does nothing");

        // The filtering drop-down opens like a combo box.
        let filtering = find(&ui, AccessibleRole::Combobox, "Filtering mode");
        assert_eq!(filtering.accessible_role(), Some(AccessibleRole::Combobox));
        assert_eq!(filtering.accessible_value().as_deref(), Some("Trilinear"));
        assert_eq!(filtering.accessible_expanded(), Some(false));
        filtering.invoke_accessible_expand_action();
        assert!(ui.get_filter_popup_visible());
        let bilinear = find(&ui, AccessibleRole::RadioButton, "Bilinear (Low Quality)");
        assert_eq!(
            bilinear.accessible_role(),
            Some(AccessibleRole::RadioButton)
        );
        assert_eq!(bilinear.accessible_checked(), Some(false));
        assert_eq!(
            find(&ui, AccessibleRole::RadioButton, "Trilinear (Balanced)").accessible_checked(),
            Some(true)
        );
        ui.hide().unwrap();
    });
}

#[test]
fn dialog_controls_act_on_their_actions() {
    crate::test_support::run_on_slint_ui_test_thread(|| {
        let ui = window();
        ui.set_export_dialog_visible(true);

        let format = find(&ui, AccessibleRole::RadioButton, "Format: PNG (Lossless)");
        assert_eq!(format.accessible_role(), Some(AccessibleRole::RadioButton));
        assert_eq!(format.accessible_checked(), Some(false));
        format.invoke_accessible_default_action();
        assert_eq!(ui.get_export_settings().format, ExportFormat::Png);
        assert_eq!(format.accessible_checked(), Some(true));

        let sharpness = find(&ui, AccessibleRole::Slider, "Sharpness");
        assert_eq!(sharpness.accessible_role(), Some(AccessibleRole::Slider));
        let before = ui.get_export_settings().sharpness;
        sharpness.invoke_accessible_increment_action();
        let after = ui.get_export_settings().sharpness;
        assert!((after - before - 0.05).abs() < 1e-4, "{before} -> {after}");
        // As typed into its value field.
        sharpness.set_accessible_value("0.25");
        assert!((ui.get_export_settings().sharpness - 0.25).abs() < 1e-4);

        let scale_bar = find(&ui, AccessibleRole::Switch, "Scale Bar");
        assert_eq!(scale_bar.accessible_role(), Some(AccessibleRole::Switch));
        let shown = ui.get_export_settings().show_scale_bar;
        scale_bar.invoke_accessible_default_action();
        assert_eq!(ui.get_export_settings().show_scale_bar, !shown);

        let exported = Rc::new(Cell::new(false));
        ui.on_export_dialog_export_requested({
            let exported = Rc::clone(&exported);
            move || exported.set(true)
        });
        find(&ui, AccessibleRole::Button, "Export").invoke_accessible_default_action();
        assert!(exported.get());
        ui.set_export_dialog_visible(false);

        // A check box drawn as a row of the dataset dialog.
        ui.set_dataset_export_dialog_visible(true);
        let tissue = find(
            &ui,
            AccessibleRole::Checkbox,
            "Tissue only (skip patches that are mostly background)",
        );
        assert_eq!(tissue.accessible_role(), Some(AccessibleRole::Checkbox));
        let before = ui.get_dataset_export_settings().tissue_only;
        tissue.invoke_accessible_default_action();
        assert_eq!(ui.get_dataset_export_settings().tissue_only, !before);
        assert_eq!(tissue.accessible_checked(), Some(!before));
        ui.hide().unwrap();
    });
}

/// Instantiates what was just shown and runs the change handlers, as the
/// event loop does between events.
fn settle(ui: &AppWindow) {
    let _ = ElementQuery::from_root(ui).find_all();
    slint::platform::update_timers_and_animations();
}

fn key(ui: &AppWindow, text: impl Into<slint::SharedString>) {
    let text = text.into();
    ui.window()
        .dispatch_event(slint::platform::WindowEvent::KeyPressed { text: text.clone() });
    ui.window()
        .dispatch_event(slint::platform::WindowEvent::KeyReleased { text });
    settle(ui);
}

fn shift_tab(ui: &AppWindow) {
    use slint::platform::Key;
    ui.window()
        .dispatch_event(slint::platform::WindowEvent::KeyPressed {
            text: Key::Shift.into(),
        });
    key(ui, Key::Tab);
    ui.window()
        .dispatch_event(slint::platform::WindowEvent::KeyReleased {
            text: Key::Shift.into(),
        });
    settle(ui);
}

/// Counts the calls of a callback.
fn counter() -> (Rc<Cell<u32>>, impl Fn() + 'static) {
    let count = Rc::new(Cell::new(0));
    let tick = {
        let count = Rc::clone(&count);
        move || count.set(count.get() + 1)
    };
    (count, tick)
}

/// A slide in the window whose view has the keyboard and, like eov's
/// interaction controller, takes every key (Escape too).
fn window_with_slide_keyboard() -> (AppWindow, Rc<Cell<u32>>) {
    let ui = window();
    fill(&ui);
    let mut pane = ui.get_panes().row_data(0).unwrap();
    pane.hud.hud_dropdown_open = false;
    // In place, as eov updates its panes (a new model would make new views).
    ui.get_panes().set_row_data(0, pane);
    settle(&ui);
    let (slide_keys, tick) = counter();
    ui.on_viewport_key(move |_, _, _, _, _, _, pressed, _| {
        if pressed {
            tick();
        }
        true
    });
    find(&ui, AccessibleRole::Image, "Slide view")
        .mock_single_click(slint::platform::PointerEventButton::Left);
    settle(&ui);
    key(&ui, "a");
    assert_eq!(slide_keys.get(), 1, "the slide has the keyboard");
    (ui, slide_keys)
}

/// Shows, reads and hides one dialog or panel.
type Surface = (&'static str, SetVisible, fn(&AppWindow) -> bool);

#[test]
fn escape_closes_dialogs_and_menus_even_when_the_slide_had_the_keyboard() {
    use slint::platform::Key;
    crate::test_support::run_on_slint_ui_test_thread(|| {
        let (ui, slide_keys) = window_with_slide_keyboard();

        // A menu opened over the slide.
        ui.set_context_menu_visible(true);
        key(&ui, Key::Escape);
        assert!(!ui.get_context_menu_visible());

        // A dialog takes the keyboard when it opens.
        let (cancelled, tick) = counter();
        ui.on_export_dialog_cancel_requested(tick);
        ui.set_export_dialog_visible(true);
        settle(&ui);
        key(&ui, Key::Escape);
        assert!(!ui.get_export_dialog_visible());
        assert_eq!(cancelled.get(), 1);

        // The window has the keyboard again (not the slide, and not
        // nothing).
        let (global_keys, tick) = counter();
        ui.on_global_key(move |_, _, _, _, _, pressed, _| {
            if pressed {
                tick();
            }
            false
        });
        let before = slide_keys.get();
        key(&ui, "a");
        assert_eq!(global_keys.get(), 1);
        assert_eq!(slide_keys.get(), before);

        // Every dialog and panel closes on Escape.
        let surfaces: [Surface; 7] = [
            (
                "dataset",
                AppWindow::set_dataset_export_dialog_visible,
                AppWindow::get_dataset_export_dialog_visible,
            ),
            (
                "slide info",
                AppWindow::set_slide_info_visible,
                AppWindow::get_slide_info_visible,
            ),
            (
                "save copy",
                AppWindow::set_save_copy_dialog_visible,
                AppWindow::get_save_copy_dialog_visible,
            ),
            (
                "plugins",
                AppWindow::set_plugins_panel_visible,
                AppWindow::get_plugins_panel_visible,
            ),
            (
                "shortcuts",
                AppWindow::set_shortcuts_panel_visible,
                AppWindow::get_shortcuts_panel_visible,
            ),
            (
                "measurements",
                AppWindow::set_measurements_panel_visible,
                AppWindow::get_measurements_panel_visible,
            ),
            (
                "lanczos",
                AppWindow::set_lanczos_confirm_visible,
                AppWindow::get_lanczos_confirm_visible,
            ),
        ];
        for (name, set_visible, visible) in surfaces {
            set_visible(&ui, true);
            settle(&ui);
            key(&ui, Key::Escape);
            assert!(!visible(&ui), "{name}");
        }
        ui.hide().unwrap();
    });
}

#[test]
fn enter_does_the_default_action() {
    use slint::platform::Key;
    crate::test_support::run_on_slint_ui_test_thread(|| {
        let ui = window();
        let (exported, tick) = counter();
        ui.on_export_dialog_export_requested(tick);
        ui.set_export_dialog_visible(true);
        settle(&ui);
        key(&ui, Key::Return);
        assert_eq!(exported.get(), 1);
        ui.set_export_dialog_visible(false);
        settle(&ui);

        let (accepted, tick) = counter();
        ui.on_lanczos_confirm_accepted(tick);
        ui.set_lanczos_confirm_visible(true);
        settle(&ui);
        key(&ui, Key::Return);
        assert_eq!(accepted.get(), 1);
        assert!(!ui.get_lanczos_confirm_visible());

        // A plugin's question is answered with Enter, unless the answer
        // destroys something.
        let (confirmed, tick) = counter();
        ui.on_plugin_confirm_accepted(tick);
        ui.set_plugin_confirm_destructive(true);
        ui.set_plugin_confirm_visible(true);
        settle(&ui);
        key(&ui, Key::Return);
        assert_eq!(confirmed.get(), 0);
        assert!(ui.get_plugin_confirm_visible());
        ui.set_plugin_confirm_visible(false);
        settle(&ui);
        ui.set_plugin_confirm_destructive(false);
        ui.set_plugin_confirm_visible(true);
        settle(&ui);
        key(&ui, Key::Return);
        assert_eq!(confirmed.get(), 1);

        // Quantify Stain: Enter in a threshold field quantifies.
        let (runs, tick) = counter();
        ui.on_quantify_run(tick);
        ui.set_quantify_visible(true);
        settle(&ui);
        key(&ui, Key::Tab);
        key(&ui, "1");
        key(&ui, Key::Return);
        assert_eq!(runs.get(), 1);
        ui.set_quantify_visible(false);
        settle(&ui);

        // A channel's value: Enter applies it, and the panel keeps the
        // keyboard (Escape still closes it).
        fill(&ui);
        let minimum = Rc::new(std::cell::RefCell::new(String::new()));
        ui.on_channels_min_entered({
            let minimum = Rc::clone(&minimum);
            move |_, text| *minimum.borrow_mut() = text.to_string()
        });
        ui.set_channels_panel_visible(true);
        settle(&ui);
        find(&ui, AccessibleRole::Spinbox, "Minimum of DAPI").invoke_accessible_default_action();
        settle(&ui);
        key(&ui, "7");
        key(&ui, Key::Return);
        assert!(minimum.borrow().ends_with('7'), "{}", minimum.borrow());
        key(&ui, Key::Escape);
        assert!(!ui.get_channels_panel_visible());
        ui.hide().unwrap();
    });
}

#[test]
fn tab_moves_through_a_dialogs_fields_without_leaving_it() {
    use slint::platform::Key;
    crate::test_support::run_on_slint_ui_test_thread(|| {
        let ui = window();
        let mut settings = ui.get_dataset_export_settings();
        settings.tile_size = 1;
        settings.stride = 1;
        settings.threads = 1;
        ui.set_dataset_export_settings(settings);
        let (exported, tick) = counter();
        ui.on_dataset_export_requested(tick);
        ui.set_dataset_export_dialog_visible(true);
        settle(&ui);

        // The dialog has the keyboard: Enter starts the export.
        key(&ui, Key::Return);
        assert_eq!(exported.get(), 1);

        // Tab goes to the tile size, the stride, the threads and the white
        // threshold, then back to the dialog and round again.
        key(&ui, Key::Tab);
        key(&ui, "2");
        assert_ne!(ui.get_dataset_export_settings().tile_size, 1, "tile size");
        key(&ui, Key::Tab);
        key(&ui, "3");
        assert_ne!(ui.get_dataset_export_settings().stride, 1, "stride");
        key(&ui, Key::Tab);
        key(&ui, "4");
        assert_ne!(ui.get_dataset_export_settings().threads, 1, "threads");
        key(&ui, Key::Tab);
        key(&ui, Key::Tab);
        // Enter on the dialog again (a field would have taken it).
        key(&ui, Key::Return);
        assert_eq!(exported.get(), 2);
        let tile_size = ui.get_dataset_export_settings().tile_size;
        key(&ui, Key::Tab);
        key(&ui, "5");
        assert_ne!(ui.get_dataset_export_settings().tile_size, tile_size);

        // Shift+Tab goes back to the dialog and stops there.
        shift_tab(&ui);
        shift_tab(&ui);
        key(&ui, Key::Return);
        assert_eq!(exported.get(), 3);
        assert!(ui.get_dataset_export_dialog_visible());
        ui.hide().unwrap();
    });
}

#[test]
fn the_slide_gets_the_keyboard_back_from_its_hud() {
    use slint::platform::Key;
    crate::test_support::run_on_slint_ui_test_thread(|| {
        let (ui, slide_keys) = window_with_slide_keyboard();

        // The zoom field: Enter applies the zoom and the slide has the
        // keyboard again.
        let zoom = Rc::new(std::cell::RefCell::new(String::new()));
        ui.on_hud_zoom_input({
            let zoom = Rc::clone(&zoom);
            move |_, text| *zoom.borrow_mut() = text.to_string()
        });
        find(&ui, AccessibleRole::Spinbox, "Zoom").invoke_accessible_default_action();
        settle(&ui);
        key(&ui, "5");
        key(&ui, Key::Return);
        assert_eq!(zoom.borrow().as_str(), "5");
        let before = slide_keys.get();
        key(&ui, "a");
        assert_eq!(slide_keys.get(), before + 1);

        // The adjustments take the keyboard; Escape closes them.
        let (toggled, tick) = counter();
        ui.on_hud_toggle_dropdown(move |_| tick());
        let set_open = |open: bool| {
            let mut pane = ui.get_panes().row_data(0).unwrap();
            pane.hud.hud_dropdown_open = open;
            ui.get_panes().set_row_data(0, pane);
            settle(&ui);
        };
        set_open(true);
        key(&ui, Key::Escape);
        assert_eq!(toggled.get(), 1);
        set_open(false);
        let before = slide_keys.get();
        key(&ui, "a");
        assert_eq!(slide_keys.get(), before + 1, "back on the slide");
        ui.hide().unwrap();
    });
}
