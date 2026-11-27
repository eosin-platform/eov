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
        let surfaces: [(&str, SetVisible); 14] = [
            ("filtering menu", AppWindow::set_filter_popup_visible),
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

/// The resource use of the running plugin in `fill`.
const USAGE: &str = "UI thread 12 ms in 340 events (0.1%)";

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
            color_management: 1,
            color_management_tooltip: "Colour management is on".into(),
            focal_plane: 1,
            focal_planes: 3,
            time_points: 1,
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
        usage: USAGE.into(),
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
    let mut export = ui.get_export_settings();
    export.has_color_profile = true;
    ui.set_export_settings(export);
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

        // A Z-stack's focal planes step from the HUD.
        let focus = find(&ui, AccessibleRole::Spinbox, "Focal plane");
        assert_eq!(focus.accessible_value().as_deref(), Some("2 / 3"));
        let steps = Rc::new(std::cell::RefCell::new(Vec::new()));
        ui.on_hud_plane_step({
            let steps = Rc::clone(&steps);
            move |pane, axis, delta| steps.borrow_mut().push((pane, axis, delta))
        });
        focus.invoke_accessible_increment_action();
        find(&ui, AccessibleRole::Button, "Previous focal plane")
            .invoke_accessible_default_action();
        assert_eq!(*steps.borrow(), [(0, 0, 1), (0, 0, -1)]);

        // The colour management chip turns the slide's profile on and off.
        let icc = find(&ui, AccessibleRole::Button, "Colour management");
        assert_eq!(icc.accessible_checkable(), Some(true));
        assert_eq!(icc.accessible_checked(), Some(true));
        assert_eq!(
            icc.accessible_description().as_deref(),
            Some("Colour management is on")
        );
        let (turned, tick) = counter();
        ui.on_hud_color_management_toggled(move |_| tick());
        icc.invoke_accessible_default_action();
        assert_eq!(turned.get(), 1);
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
        // The pixel view is a check box in the same menu.
        let pixels = find(&ui, AccessibleRole::Checkbox, "Show Pixels Above 100%");
        assert_eq!(pixels.accessible_checkable(), Some(true));
        assert_eq!(pixels.accessible_checked(), Some(false));
        let (switched, tick) = counter();
        ui.on_pixel_view_toggled(tick);
        pixels.invoke_accessible_default_action();
        assert_eq!(switched.get(), 1);
        assert!(!ui.get_filter_popup_visible(), "choosing closes the menu");
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
        // A decimal comma is read in every language; what is not a number
        // leaves the value as it is.
        sharpness.set_accessible_value("0,5");
        assert!((ui.get_export_settings().sharpness - 0.5).abs() < 1e-4);
        sharpness.set_accessible_value("abc");
        assert!((ui.get_export_settings().sharpness - 0.5).abs() < 1e-4);

        // Offered for slides with a colour profile.
        let mut settings = ui.get_export_settings();
        settings.has_color_profile = true;
        ui.set_export_settings(settings);
        let icc = find(&ui, AccessibleRole::Switch, "Convert to sRGB (ICC profile)");
        let converted = ui.get_export_settings().color_management;
        icc.invoke_accessible_default_action();
        assert_eq!(ui.get_export_settings().color_management, !converted);
        assert_eq!(icc.accessible_checked(), Some(!converted));

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

/// Types `text` into the field `element`, replacing what it holds, one key
/// at a time as a keyboard does: each character goes through the field's
/// input filter (an `InputType.number` field drops all but ASCII digits).
fn type_into(ui: &AppWindow, element: &ElementHandle, text: &str) {
    use slint::platform::{Key, WindowEvent};
    element.mock_single_click(slint::platform::PointerEventButton::Left);
    settle(ui);
    ui.window().dispatch_event(WindowEvent::KeyPressed {
        text: Key::Control.into(),
    });
    key(ui, "a");
    ui.window().dispatch_event(WindowEvent::KeyReleased {
        text: Key::Control.into(),
    });
    for character in text.chars() {
        key(ui, character.to_string());
    }
}

/// The dialogs' number fields take the digits Persian and Arabic keyboards
/// type, and what is not a number changes nothing (Enter shows the number
/// in use again); whole numbers stay whole, also when they are written with
/// a thousands separator, and count as they are typed.
#[test]
fn number_fields_take_the_digits_every_keyboard_types() {
    use slint::platform::Key;
    crate::test_support::run_on_slint_ui_test_thread(|| {
        let ui = window();

        ui.set_dataset_export_settings(crate::DatasetExportSettings {
            tile_size: 256,
            stride: 256,
            threads: 4,
            ..Default::default()
        });
        ui.set_dataset_export_dialog_visible(true);
        settle(&ui);
        let tile_size = find(&ui, AccessibleRole::TextInput, "Patch size");
        // Persian digits, read as they are typed.
        type_into(&ui, &tile_size, "۵۱۲");
        assert_eq!(ui.get_dataset_export_settings().tile_size, 512);
        // Not a number: the patch size stays, and Enter shows it again.
        type_into(&ui, &tile_size, "x");
        assert_eq!(ui.get_dataset_export_settings().tile_size, 512);
        assert_eq!(tile_size.accessible_value().as_deref(), Some("x"));
        key(&ui, Key::Return);
        assert_eq!(tile_size.accessible_value().as_deref(), Some("512"));
        // Arabic-Indic digits and a decimal: a whole number of pixels.
        let stride = find(&ui, AccessibleRole::TextInput, "Stride");
        type_into(&ui, &stride, "١٢٨٫٦");
        key(&ui, Key::Return);
        assert_eq!(ui.get_dataset_export_settings().stride, 129);
        assert_eq!(stride.accessible_value().as_deref(), Some("129"));
        let threads = find(&ui, AccessibleRole::TextInput, "Threads");
        type_into(&ui, &threads, "inf");
        key(&ui, Key::Return);
        assert_eq!(ui.get_dataset_export_settings().threads, 4);
        // A thousands separator, as English, German, French or Persian
        // write one: the whole number counts as soon as it is typed (the
        // "1" typed first is not what stays), with no Enter.
        type_into(&ui, &stride, "1,024");
        assert_eq!(ui.get_dataset_export_settings().stride, 1024);
        assert_eq!(stride.accessible_value().as_deref(), Some("1,024"));
        for grouped in [
            "1.024",
            "1 024",
            "1\u{A0}024",
            "1\u{202F}024",
            "١٬٠٢٤",
            "۱٬۰۲۴",
        ] {
            type_into(&ui, &stride, "512");
            assert_eq!(ui.get_dataset_export_settings().stride, 512);
            type_into(&ui, &stride, grouped);
            assert_eq!(ui.get_dataset_export_settings().stride, 1024, "{grouped:?}");
        }
        key(&ui, Key::Return);
        assert_eq!(stride.accessible_value().as_deref(), Some("1024"));
        type_into(&ui, &tile_size, "8,192");
        assert_eq!(ui.get_dataset_export_settings().tile_size, 8192);
        type_into(&ui, &threads, "1.000");
        assert_eq!(ui.get_dataset_export_settings().threads, 1000);
        // Other decimals are rounded, also with a decimal comma.
        type_into(&ui, &threads, "2,5");
        assert_eq!(ui.get_dataset_export_settings().threads, 3);
        let white = find(&ui, AccessibleRole::TextInput, "White threshold (%)");
        type_into(&ui, &white, "۴۰");
        assert!((ui.get_dataset_export_settings().white_threshold - 0.4).abs() < 1e-6);
        ui.set_dataset_export_dialog_visible(false);

        let mut settings = ui.get_export_settings();
        settings.has_measurement = true;
        settings.show_measurement = true;
        settings.measurement_color = slint::Color::from_rgb_u8(10, 20, 30);
        settings.measurement_opacity = 80.0;
        settings.measurement_stroke_style = crate::StrokeStyle::Dashed;
        settings.measurement_dash_length = 8.0;
        ui.set_export_settings(settings);
        ui.set_export_dialog_visible(true);
        settle(&ui);
        // An empty field is not 0 %.
        let opacity = find(&ui, AccessibleRole::TextInput, "Opacity (%)");
        type_into(&ui, &opacity, "");
        key(&ui, Key::Backspace);
        assert_eq!(opacity.accessible_value().as_deref(), Some(""));
        key(&ui, Key::Return);
        assert_eq!(ui.get_export_settings().measurement_opacity, 80.0);
        assert_eq!(opacity.accessible_value().as_deref(), Some("80"));
        type_into(&ui, &opacity, "۵۰");
        key(&ui, Key::Return);
        assert_eq!(ui.get_export_settings().measurement_opacity, 50.0);
        // A dash length of 0 would leave no dashes.
        let dash = find(&ui, AccessibleRole::TextInput, "Dash length");
        type_into(&ui, &dash, "0");
        key(&ui, Key::Return);
        assert_eq!(ui.get_export_settings().measurement_dash_length, 8.0);
        type_into(&ui, &dash, "۱۲");
        key(&ui, Key::Return);
        assert_eq!(ui.get_export_settings().measurement_dash_length, 12.0);
        // The colour's red, green and blue (in the colour menu).
        find(&ui, AccessibleRole::Button, "Colour").invoke_accessible_default_action();
        settle(&ui);
        let red = find(&ui, AccessibleRole::TextInput, "Red (0–255)");
        type_into(&ui, &red, "۲۰۰");
        key(&ui, Key::Return);
        assert_eq!(ui.get_export_settings().measurement_color.red(), 200);
        type_into(&ui, &red, "abc");
        key(&ui, Key::Return);
        assert_eq!(ui.get_export_settings().measurement_color.red(), 200);
        assert_eq!(red.accessible_value().as_deref(), Some("200"));
        ui.set_export_dialog_visible(false);
        ui.hide().unwrap();
    });
}

/// The colour menu's red, green and blue fields use a value as it is
/// typed: moving on to the next field (Tab, a click) or setting it through
/// assistive technology keeps it, and Enter shows the value used.
#[test]
fn colour_values_count_without_enter() {
    use slint::platform::Key;
    crate::test_support::run_on_slint_ui_test_thread(|| {
        let ui = window();
        let mut settings = ui.get_export_settings();
        settings.has_measurement = true;
        settings.show_measurement = true;
        settings.measurement_color = slint::Color::from_rgb_u8(46, 204, 113);
        ui.set_export_settings(settings);
        ui.set_export_dialog_visible(true);
        settle(&ui);
        find(&ui, AccessibleRole::Button, "Colour").invoke_accessible_default_action();
        settle(&ui);
        let rgb = |ui: &AppWindow| {
            let color = ui.get_export_settings().measurement_color;
            (color.red(), color.green(), color.blue())
        };
        let red = find(&ui, AccessibleRole::TextInput, "Red (0–255)");
        let green = find(&ui, AccessibleRole::TextInput, "Green (0–255)");
        let blue = find(&ui, AccessibleRole::TextInput, "Blue (0–255)");

        // Red, Tab, green, Tab, blue, Enter: all three are used. Tab
        // selects the next field's text, so typing replaces it.
        type_into(&ui, &red, "255");
        key(&ui, Key::Tab);
        key(&ui, "0");
        key(&ui, Key::Tab);
        key(&ui, "0");
        key(&ui, Key::Return);
        assert_eq!(rgb(&ui), (255, 0, 0));
        assert_eq!(red.accessible_value().as_deref(), Some("255"));
        assert_eq!(green.accessible_value().as_deref(), Some("0"));

        // A click into the next field instead, and no Enter at all.
        type_into(&ui, &red, "10");
        type_into(&ui, &green, "20");
        assert_eq!(rgb(&ui), (10, 20, 0));

        // While a value is typed, the field keeps what was typed; Enter
        // shows the value used (at most 255).
        type_into(&ui, &blue, "300");
        assert_eq!(rgb(&ui), (10, 20, 255));
        assert_eq!(blue.accessible_value().as_deref(), Some("300"));
        key(&ui, Key::Return);
        assert_eq!(blue.accessible_value().as_deref(), Some("255"));

        // Set by assistive technology (Slint's default: the text, then
        // `edited`).
        blue.set_accessible_value("77");
        settle(&ui);
        assert_eq!(rgb(&ui), (10, 20, 77));
        // What is not a number leaves the colour as it is.
        green.set_accessible_value("abc");
        settle(&ui);
        assert_eq!(rgb(&ui), (10, 20, 77));
        ui.set_export_dialog_visible(false);
        ui.hide().unwrap();
    });
}

#[test]
fn a_running_plugin_shows_what_it_costs() {
    crate::test_support::run_on_slint_ui_test_thread(|| {
        let ui = window();
        fill(&ui);
        ui.set_plugins_panel_visible(true);
        let usage = find(&ui, AccessibleRole::Text, USAGE);
        assert_eq!(
            usage.accessible_description().as_deref(),
            Some("Resources used since the plugin started")
        );
        // A plugin that does not run has no such line.
        let rows = ui.get_plugin_rows();
        let mut row = rows.row_data(0).unwrap();
        row.usage = "".into();
        rows.set_row_data(0, row);
        settle(&ui);
        assert_eq!(
            ElementHandle::find_by_accessible_label(&ui, USAGE).count(),
            0
        );
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

/// A key the slide takes is not handed to the window's shortcuts too (it
/// ran twice: `]` turned the view 180°, `=` zoomed two steps).
#[test]
fn a_key_the_slide_takes_runs_once() {
    crate::test_support::run_on_slint_ui_test_thread(|| {
        let (ui, slide_keys) = window_with_slide_keyboard();
        let (global_keys, tick) = counter();
        ui.on_global_key(move |_, _, _, _, _, pressed, _| {
            if pressed {
                tick();
            }
            true
        });
        let before = slide_keys.get();
        key(&ui, "]");
        assert_eq!(slide_keys.get(), before + 1);
        assert_eq!(global_keys.get(), 0, "the shortcut ran a second time");
        ui.hide().unwrap();
    });
}

/// Escape closes the confirmation over the Plugins panel, not the panel;
/// the panel has the keyboard again.
#[test]
fn escape_closes_the_surface_on_top_only() {
    use slint::platform::Key;
    crate::test_support::run_on_slint_ui_test_thread(|| {
        let ui = window();
        let (cancelled, tick) = counter();
        ui.on_plugin_confirm_cancelled(tick);
        ui.set_plugins_panel_visible(true);
        settle(&ui);
        ui.set_plugin_confirm_visible(true);
        settle(&ui);
        key(&ui, Key::Escape);
        assert_eq!(cancelled.get(), 1);
        assert!(!ui.get_plugin_confirm_visible());
        assert!(ui.get_plugins_panel_visible());
        key(&ui, Key::Escape);
        assert!(!ui.get_plugins_panel_visible());
        ui.hide().unwrap();
    });
}

/// While a dialog or panel over the window is open, the window's shortcuts
/// do not run behind it (UI-7).
#[test]
fn shortcuts_do_not_run_behind_a_dialog() {
    crate::test_support::run_on_slint_ui_test_thread(|| {
        let ui = window();
        let (global_keys, tick) = counter();
        ui.on_global_key(move |_, _, _, _, _, pressed, _| {
            if pressed {
                tick();
            }
            true
        });
        key(&ui, "m");
        assert_eq!(global_keys.get(), 1, "with no dialog open");
        ui.set_measurements_panel_visible(true);
        settle(&ui);
        key(&ui, "m");
        assert_eq!(global_keys.get(), 1);
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

/// A window at its narrowest (800 px wide) with `count` plugin buttons in
/// the toolbar, eight per plugin, the last of each a plain button.
fn narrow_window_with_plugin_buttons(count: usize) -> AppWindow {
    let ui = AppWindow::new().unwrap();
    ui.window().set_size(slint::PhysicalSize::new(800, 600));
    ui.show().unwrap();
    let buttons = (0..count)
        .map(|i| crate::PluginButtonData {
            plugin_id: format!("plugin-{}", i / 8).into(),
            button_id: format!("button-{i}").into(),
            tooltip: format!("Plugin button {i}").into(),
            enabled: true,
            is_tool: i % 8 != 7,
            show_left_separator: i % 8 == 0,
            ..Default::default()
        })
        .collect();
    ui.set_plugin_buttons(model(buttons));
    settle(&ui);
    ui
}

/// Where an element is in the window: (left, top, right, bottom).
fn bounds(element: &ElementHandle) -> (f32, f32, f32, f32) {
    let (position, size) = (element.absolute_position(), element.size());
    (
        position.x,
        position.y,
        position.x + size.width,
        position.y + size.height,
    )
}

fn center(bounds: (f32, f32, f32, f32)) -> (f32, f32) {
    ((bounds.0 + bounds.2) / 2.0, (bounds.1 + bounds.3) / 2.0)
}

fn title_bar_height(ui: &AppWindow) -> f32 {
    ElementHandle::find_by_element_type_name(ui, "WindowToolbar")
        .next()
        .unwrap()
        .size()
        .height
}

fn logo(ui: &AppWindow) -> (f32, f32, f32, f32) {
    bounds(
        &ElementHandle::find_by_element_id(ui, "WindowToolbar::logo")
            .next()
            .expect("the title bar's logo"),
    )
}

#[test]
fn the_title_bar_wraps_its_buttons_and_keeps_the_logo_and_window_buttons_in_view() {
    crate::test_support::run_on_slint_ui_test_thread(|| {
        let ui = window();
        let one_row = title_bar_height(&ui);
        ui.hide().unwrap();

        let ui = narrow_window_with_plugin_buttons(24);
        // The window's own buttons stay in the top right corner, the logo
        // beside them.
        let mut right = 800.0;
        for label in ["Close window", "Maximize window", "Minimize window"] {
            let (left, top, end, _) = bounds(&find(&ui, AccessibleRole::Button, label));
            assert_eq!((top, end), (0.0, right), "{label}");
            right = left;
        }
        let logo = logo(&ui);
        assert_eq!((logo.1, logo.2), (0.0, right), "the logo");
        assert!(logo.0 < logo.2);

        // What does not fit beside the first row wraps onto rows below it,
        // left of the logo, each button after the one before.
        let first_row = bounds(&find(&ui, AccessibleRole::Button, "Toggle minimap")).1;
        let height = title_bar_height(&ui);
        let buttons: Vec<_> = (0..24)
            .map(|i| {
                bounds(&find(
                    &ui,
                    AccessibleRole::Button,
                    &format!("Plugin button {i}"),
                ))
            })
            .collect();
        let mut rows: Vec<f32> = buttons.iter().map(|button| button.1).collect();
        rows.dedup();
        assert!(rows.len() >= 2, "rows at {rows:?}");
        assert!(rows.iter().all(|&top| top > first_row), "rows at {rows:?}");
        for (i, &(left, _, end, bottom)) in buttons.iter().enumerate() {
            assert!(
                left >= 0.0 && end <= logo.0 && bottom <= height,
                "button {i} at {:?}",
                buttons[i]
            );
        }
        for pair in buttons.windows(2) {
            let (before, after) = (pair[0], pair[1]);
            assert!(
                after.1 > before.1 || after.0 >= before.2,
                "{before:?} then {after:?}"
            );
        }
        let plugins = bounds(&find(&ui, AccessibleRole::Button, "Plugins (0)"));
        assert!(plugins.2 <= logo.0 && plugins.3 <= height, "{plugins:?}");
        // The title bar grows by a row of buttons for each row.
        assert!(
            height >= one_row + rows.len() as f32 * 28.0,
            "{one_row} -> {height}"
        );

        // A button on a wrapped row works.
        let clicked = Rc::new(std::cell::RefCell::new(None));
        ui.on_plugin_button_clicked({
            let clicked = Rc::clone(&clicked);
            move |plugin, button, is_tool| {
                *clicked.borrow_mut() = Some((plugin.to_string(), button.to_string(), is_tool));
            }
        });
        find(&ui, AccessibleRole::Button, "Plugin button 23")
            .mock_single_click(slint::platform::PointerEventButton::Left);
        assert_eq!(
            clicked.borrow().clone(),
            Some(("plugin-2".to_string(), "button-23".to_string(), false))
        );
        ui.hide().unwrap();
    });
}

#[test]
fn the_title_bars_empty_space_moves_and_maximizes_the_window() {
    use slint::platform::{PointerEventButton, WindowEvent};
    crate::test_support::run_on_slint_ui_test_thread(|| {
        let ui = narrow_window_with_plugin_buttons(24);
        let (maximized, tick) = counter();
        ui.on_window_maximize(tick);
        // A click; whether its press would move the window (callbacks.rs
        // starts the move when the pointer then moves).
        let click = |(x, y): (f32, f32)| {
            let position = slint::LogicalPosition::new(x, y);
            let button = PointerEventButton::Left;
            ui.window()
                .dispatch_event(WindowEvent::PointerMoved { position });
            ui.window()
                .dispatch_event(WindowEvent::PointerPressed { position, button });
            let moves_window = ui.get_title_bar_drag_pressed();
            ui.window()
                .dispatch_event(WindowEvent::PointerReleased { position, button });
            moves_window
        };
        // Long enough after the last click not to make a double click.
        let pause = || {
            i_slint_backend_testing::mock_elapsed_time(std::time::Duration::from_secs(1));
            settle(&ui);
        };
        let press = |at: (f32, f32)| {
            let moves_window = click(at);
            pause();
            moves_window
        };
        let double_click = |at: (f32, f32)| {
            click(at);
            click(at);
            pause();
        };

        let minimize = bounds(&find(&ui, AccessibleRole::Button, "Minimize window"));
        let maximize = bounds(&find(&ui, AccessibleRole::Button, "Maximize window"));
        let logo = logo(&ui);
        let undo = bounds(&find(&ui, AccessibleRole::Button, "Undo"));
        let redo = bounds(&find(&ui, AccessibleRole::Button, "Redo"));
        let minimap = bounds(&find(&ui, AccessibleRole::Button, "Toggle minimap"));
        let dataset = bounds(&find(&ui, AccessibleRole::Button, "Export Dataset"));
        let height = title_bar_height(&ui);

        // The strip along the top, from the left edge to the window's
        // buttons, down to the first row of buttons.
        for x in [1.0, 400.0, minimize.0 - 1.0] {
            assert!(press((x, 1.0)), "top edge at x {x}");
            assert!(press((x, minimap.1 - 1.0)), "above the buttons at x {x}");
        }
        // The logo, a gap between two buttons, the rest of the first row and
        // the bottom edge.
        assert!(press(center(logo)), "logo");
        assert!(
            press(((undo.2 + redo.0) / 2.0, center(undo).1)),
            "between buttons"
        );
        assert!(
            press(((dataset.2 + logo.0) / 2.0, center(dataset).1)),
            "after the first row"
        );
        assert!(press((400.0, height - 2.0)), "bottom edge");
        // Not the buttons, disabled ones included, nor the window's buttons
        // where the strip meets them, nor the slides below.
        assert!(!press(center(minimap)), "a button");
        assert!(!press(center(undo)), "a disabled button");
        assert!(!press((center(minimize).0, 1.0)), "a window button");
        assert!(!press((400.0, height + 10.0)), "below the title bar");
        assert_eq!(maximized.get(), 0);

        // A double click there maximizes or restores the window, as the
        // window's maximize button does at the top of the title bar.
        double_click((400.0, 1.0));
        assert_eq!(maximized.get(), 1, "the strip");
        double_click(center(logo));
        assert_eq!(maximized.get(), 2, "the logo");
        double_click(center(undo));
        assert_eq!(maximized.get(), 2, "a disabled button");
        press((center(maximize).0, 1.0));
        assert_eq!(maximized.get(), 3, "the maximize button");
        ui.hide().unwrap();
    });
}

#[test]
fn with_the_systems_window_buttons_the_title_bar_leaves_them_room() {
    crate::test_support::run_on_slint_ui_test_thread(|| {
        let ui = narrow_window_with_plugin_buttons(24);
        ui.set_use_native_window_controls(true);
        settle(&ui);
        // macOS draws its window buttons at the top left: the toolbar starts
        // after them, its first row where they are aligned to it (40 px).
        let open = bounds(&find(
            &ui,
            AccessibleRole::Button,
            "Open file (right-click for folder or recent)",
        ));
        assert_eq!((open.0, open.1, open.3), (84.0, 6.0, 34.0));
        // eov's own window buttons take no room: the logo ends the first row.
        let logo = logo(&ui);
        assert_eq!(logo.2, 800.0);
        // The rest still wraps, rows below the first starting at the left.
        let wrapped = bounds(&find(&ui, AccessibleRole::Button, "Plugin button 23"));
        assert!(wrapped.1 > open.1 && wrapped.2 <= logo.0, "{wrapped:?}");
        let render = bounds(&find(&ui, AccessibleRole::Button, "Use CPU renderer"));
        assert!(render.1 > open.1 && render.0 < 20.0, "{render:?}");
        ui.hide().unwrap();
    });
}

/// Back to English when a test ends, also when it fails.
struct English<'a>(&'a AppWindow);

impl Drop for English<'_> {
    fn drop(&mut self) {
        crate::i18n::switch_to(self.0, eov_plugin_api::i18n::source());
    }
}

/// The elements with this id (`Component::id`) that are on screen.
fn elements(ui: &AppWindow, id: &str) -> Vec<ElementHandle> {
    let found: Vec<_> = ElementHandle::find_by_element_id(ui, id).collect();
    assert!(!found.is_empty(), "no element {id}");
    found
}

/// Whether `inner` lies within `outer` (both as [`bounds`] gives them).
fn inside(inner: (f32, f32, f32, f32), outer: (f32, f32, f32, f32)) -> bool {
    const SLACK: f32 = 0.5;
    inner.0 >= outer.0 - SLACK
        && inner.1 >= outer.1 - SLACK
        && inner.2 <= outer.2 + SLACK
        && inner.3 <= outer.3 + SLACK
}

/// Polls `future` to its end (compiling with the interpreter waits for
/// nothing).
fn spin_on<T>(future: impl std::future::Future<Output = T>) -> T {
    use std::task::{Context, Poll, Waker};
    let mut cx = Context::from_waker(Waker::noop());
    let mut future = std::pin::pin!(future);
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut cx) {
            return value;
        }
    }
}

/// Measures texts as the window lays them out, in the same (the system's)
/// fonts: what a text needs, to compare with what its element is given.
struct TextMeasure(slint_interpreter::ComponentInstance);

impl TextMeasure {
    fn new() -> Self {
        let source = r#"
            export component Measure inherits Window {
                in property <string> text;
                in property <length> size;
                in property <length> wrap-width;
                out property <length> line-width: line.preferred-width;
                out property <length> wrapped-height: wrapped.preferred-height;

                line := Text {
                    text: root.text;
                    font-size: root.size;
                }

                wrapped := Text {
                    width: root.wrap-width;
                    text: root.text;
                    font-size: root.size;
                    wrap: word-wrap;
                }
            }
        "#;
        let result = spin_on(
            slint_interpreter::Compiler::default()
                .build_from_source(source.into(), "measure.slint".into()),
        );
        let definition = result
            .component("Measure")
            .unwrap_or_else(|| panic!("{:?}", result.diagnostics().collect::<Vec<_>>()));
        Self(definition.create().unwrap())
    }

    fn measure(&self, text: &str, size: f32, wrap_width: f32, what: &str) -> f32 {
        use slint_interpreter::Value;
        let set = |name: &str, value: Value| self.0.set_property(name, value).unwrap();
        set("text", Value::String(text.into()));
        set("size", Value::Number(size.into()));
        set("wrap-width", Value::Number(wrap_width.into()));
        match self.0.get_property(what).unwrap() {
            Value::Number(length) => length as f32,
            other => panic!("{what}: {other:?}"),
        }
    }

    /// The width of `text` on one line, at `size`.
    fn width(&self, text: &str, size: f32) -> f32 {
        self.measure(text, size, 0.0, "line-width")
    }

    /// The height of `text` at `size`, word-wrapped at `width`.
    fn wrapped_height(&self, text: &str, size: f32, width: f32) -> f32 {
        self.measure(text, size, width, "wrapped-height")
    }
}

/// Whether two lengths are the same, but for rounding.
fn same(a: f32, b: f32) -> bool {
    (a - b).abs() < 0.5
}

/// The Dataset export dialog's checkbox labels wrap inside the dialog in
/// every language (Russian ran past its border) and are shown in full: each
/// row is as tall as before (22 px), taller by what its wrapped label needs,
/// and the rows do not overlap. A click anywhere on a row toggles its
/// checkbox (the label too, not only the space after it).
#[test]
fn dataset_checkbox_labels_wrap_inside_the_dialog() {
    use slint::platform::PointerEventButton;
    crate::test_support::run_on_slint_ui_test_thread(|| {
        let ui = window();
        let _english = English(&ui);
        let measure = TextMeasure::new();
        ui.set_dataset_export_dialog_visible(true);
        settle(&ui);
        // Labels that take more than one line (in several languages).
        let mut wrapped = 0;
        for language in eov_plugin_api::i18n::LANGUAGES {
            crate::i18n::switch_to(&ui, language);
            settle(&ui);
            let tag = language.tag;
            let size = ui.global::<crate::Locale>().invoke_text_size(11.0);
            let card = bounds(&elements(&ui, "DatasetExportDialog::card")[0]);
            // The rows (the checkboxes) and their labels, from the top.
            let mut rows: Vec<_> = ElementHandle::find_by_element_type_name(&ui, "CheckRow")
                .map(|row| (bounds(&row), row.accessible_label().unwrap()))
                .collect();
            rows.sort_by(|a, b| a.0.1.total_cmp(&b.0.1));
            let mut labels: Vec<_> = elements(&ui, "CheckRow::label")
                .iter()
                .map(bounds)
                .collect();
            labels.sort_by(|a, b| a.1.total_cmp(&b.1));
            assert_eq!((rows.len(), labels.len()), (3, 3), "{tag}");
            for ((row, text), &at) in rows.iter().zip(&labels) {
                assert!(
                    inside(at, card),
                    "{tag}: {text:?} at {at:?} is not inside the dialog {card:?}"
                );
                // All of it: its lines at the width it has, inside its row.
                let needed = measure.wrapped_height(text, size, at.2 - at.0);
                assert!(
                    at.3 - at.1 >= needed - 0.5 && inside(at, *row),
                    "{tag}: {text:?} at {at:?} needs {needed} px, in the row {row:?}"
                );
                assert!(
                    same(row.3 - row.1, 22f32.max(needed + 5.0)),
                    "{tag}: the row of {text:?} is {row:?}, its label needs {needed} px"
                );
                if needed > measure.wrapped_height(text, size, 10000.0) + 0.5 {
                    wrapped += 1;
                }
            }
            for pair in rows.windows(2) {
                assert!(pair[0].0.3 <= pair[1].0.1, "{tag}: rows overlap: {pair:?}");
            }
        }
        assert!(
            wrapped > 0,
            "no label took two lines: rows that grow went untested"
        );
        crate::i18n::switch_to(&ui, eov_plugin_api::i18n::source());
        settle(&ui);
        let before = ui.get_dataset_export_settings();
        for label in elements(&ui, "CheckRow::label") {
            label.mock_single_click(PointerEventButton::Left);
            settle(&ui);
        }
        let after = ui.get_dataset_export_settings();
        assert_eq!(
            (after.tissue_only, after.exclude_qc, after.deidentify),
            (!before.tissue_only, !before.exclude_qc, !before.deidentify)
        );
        ui.set_dataset_export_dialog_visible(false);
        ui.hide().unwrap();
    });
}

/// The Save a Copy dialog's choices wrap their details inside the dialog
/// in every language (the OME-TIFF one hid what OpenSlide cannot see, in
/// English too): a long detail makes its choice taller, and the dialog
/// holds it; short ones keep the height they had.
#[test]
fn save_copy_details_wrap_inside_the_dialog() {
    crate::test_support::run_on_slint_ui_test_thread(|| {
        let ui = window();
        let _english = English(&ui);
        let mut settings = ui.get_save_copy_settings();
        settings.title = "C3L-00004-21.svs".into();
        settings.whole_text = "25895 × 23396 px".into();
        settings.summary = "25895 × 23396 px in 7 levels of 512-px tiles at 0.4942 µm/px.".into();
        settings.deidentify = true;
        ui.set_save_copy_settings(settings.clone());
        ui.set_save_copy_dialog_visible(true);
        settle(&ui);
        for language in eov_plugin_api::i18n::LANGUAGES {
            crate::i18n::switch_to(&ui, language);
            settle(&ui);
            let card = bounds(&elements(&ui, "SaveCopyDialog::card")[0]);
            for detail in elements(&ui, "Choice::detail") {
                assert!(
                    inside(bounds(&detail), card),
                    "{}: {:?} is not inside the dialog {card:?}",
                    language.tag,
                    bounds(&detail)
                );
            }
            for label in elements(&ui, "CheckRow::label") {
                assert!(inside(bounds(&label), card), "{}", language.tag);
            }
        }
        crate::i18n::switch_to(&ui, eov_plugin_api::i18n::source());
        settle(&ui);
        let whole = find(&ui, AccessibleRole::RadioButton, "Whole slide");
        assert_eq!(whole.size().height, 38.0, "a one-line detail as before");
        let jpeg = find(&ui, AccessibleRole::RadioButton, "JPEG, high quality (90)");
        assert_eq!(jpeg.size().height, 24.0, "no detail, as before");
        settings.has_roi = true;
        settings.roi_text = "1234 × 5678 px at 9, 10; ".repeat(24).into();
        ui.set_save_copy_settings(settings);
        settle(&ui);
        let roi = find(&ui, AccessibleRole::RadioButton, "Region of interest");
        assert!(roi.size().height > 76.0, "{:?}", roi.size());
        let card = bounds(&elements(&ui, "SaveCopyDialog::card")[0]);
        assert!(inside(bounds(&roi), card), "{:?} {card:?}", bounds(&roi));
        let save = bounds(&find(&ui, AccessibleRole::Button, "Save…"));
        assert!(save.1 > bounds(&roi).3 && inside(save, card), "{save:?}");
        ui.set_save_copy_dialog_visible(false);
        ui.hide().unwrap();
    });
}

/// The export dialog's stroke and caps rows: in every language each label
/// gets the width its text needs and a gap of 8 px, in one column for both
/// rows, and the choices one width that fits the longest of their texts
/// with 6 px on either side; the rows fit in the dialog. Where the texts
/// need less (English), the rows look as before: 46 px labels, 52 px
/// choices.
#[test]
fn export_line_style_rows_fit_their_texts() {
    crate::test_support::run_on_slint_ui_test_thread(|| {
        let ui = window();
        let _english = English(&ui);
        let measure = TextMeasure::new();
        let mut settings = ui.get_export_settings();
        settings.has_measurement = true;
        settings.show_measurement = true;
        // The widest stroke row: with the dash length and gap.
        settings.measurement_stroke_style = crate::StrokeStyle::Dashed;
        settings.has_roi = true;
        settings.show_roi_outline = true;
        ui.set_export_settings(settings);
        ui.set_export_dialog_visible(true);
        settle(&ui);
        // Languages whose texts need more than 46 px labels and 52 px
        // choices (French "Extrémités", the Russian choices).
        let (mut wider_labels, mut wider_choices) = (0, 0);
        for language in eov_plugin_api::i18n::LANGUAGES {
            crate::i18n::switch_to(&ui, language);
            settle(&ui);
            let tag = language.tag;
            let size = ui.global::<crate::Locale>().invoke_text_size(11.0);
            let card = bounds(&elements(&ui, "ExportDialog::card")[0]);
            let strokes = elements(&ui, "StrokeStyleSelector::label");
            let caps = elements(&ui, "CapStyleSelector::label");
            assert_eq!((strokes.len(), caps.len()), (2, 2), "{tag}");
            // The choices' texts, as wide as they are (centred in their
            // choices).
            let texts: Vec<_> = elements(&ui, "SegmentButton::label")
                .iter()
                .map(bounds)
                .collect();
            for (stroke_label, caps_label) in strokes.iter().zip(&caps) {
                // One column for both rows, as wide as the longer label
                // needs with its gap, and 46 px at least.
                let needed = |label: &ElementHandle| {
                    measure.width(&label.accessible_label().unwrap(), size) + 8.0
                };
                let column = 46f32.max(needed(stroke_label).max(needed(caps_label)));
                wider_labels += usize::from(column > 46.0);
                let rows = [bounds(stroke_label), bounds(caps_label)];
                assert_eq!(rows[0].2, rows[1].2, "{tag}: one label column");
                for (label, row) in [stroke_label, caps_label].iter().zip(rows) {
                    assert!(
                        same(row.2 - row.0, column),
                        "{tag}: {:?} is {} px wide, its column needs {column} px",
                        label.accessible_label(),
                        row.2 - row.0
                    );
                }
                // The choices (and fields) after each label.
                let mut columns = Vec::new();
                for label in rows {
                    let choices: Vec<_> = ElementQuery::from_root(&ui)
                        .match_accessible_role(AccessibleRole::RadioButton)
                        .find_all()
                        .into_iter()
                        .map(|choice| bounds(&choice))
                        .filter(|choice| {
                            (choice.1 + choice.3) / 2.0 > label.1
                                && (choice.1 + choice.3) / 2.0 < label.3
                        })
                        .collect();
                    assert_eq!(choices.len(), 3, "{tag}: {choices:?}");
                    let mut right = label.2;
                    for choice in &choices {
                        assert!(
                            choice.0 >= right,
                            "{tag}: {choice:?} overlaps what comes before it"
                        );
                        assert!(inside(*choice, card), "{tag}: {choice:?} outside {card:?}");
                        right = choice.2;
                    }
                    columns.push(choices);
                }
                for (a, b) in columns[0].iter().zip(&columns[1]) {
                    assert_eq!((a.0, a.2), (b.0, b.2), "{tag}: the choices line up");
                }
                // Each choice holds its text with 6 px on either side, and
                // all six are as wide as the longest text needs (52 px at
                // least).
                let mut widest = 0f32;
                for choice in columns.iter().flatten() {
                    let text = texts
                        .iter()
                        .find(|text| {
                            let (x, y) = center(**text);
                            x > choice.0 && x < choice.2 && y > choice.1 && y < choice.3
                        })
                        .unwrap_or_else(|| panic!("{tag}: no text in the choice {choice:?}"));
                    assert!(
                        text.2 - text.0 + 12.0 <= choice.2 - choice.0 + 0.5
                            && inside(*text, *choice),
                        "{tag}: the text {text:?} does not fit in its choice {choice:?}"
                    );
                    widest = widest.max(text.2 - text.0);
                }
                let width = 52f32.max(widest + 12.0);
                wider_choices += usize::from(width > 52.0);
                for choice in columns.iter().flatten() {
                    assert!(
                        same(choice.2 - choice.0, width),
                        "{tag}: {choice:?} is not {width} px wide"
                    );
                }
            }
            // The dash length and gap fields fit too.
            for field in ElementHandle::find_by_element_type_name(&ui, "SmallTextInput") {
                assert!(inside(bounds(&field), card), "{tag}: {:?}", bounds(&field));
            }
        }
        assert!(
            wider_labels > 0 && wider_choices > 0,
            "no text needed more room ({wider_labels}, {wider_choices}): wider rows went untested"
        );
        ui.set_export_dialog_visible(false);
        ui.hide().unwrap();
    });
}

/// The HUD's Image Adjustments: the label column is as wide as its longest
/// label needs, 68 px at least, and the panel (260 px) grows with it. A
/// slider's label also ends 7 px before the slider's handle at 0 (the
/// Japanese "Sharpness" touched it); the Units and Stain Norm labels take
/// the whole column, 4 px before their buttons. So the English panel
/// keeps its size with any font that fits its labels in 60 and 68 px. The
/// six Units and Stain Norm choices are a third of 160 px wide, or as wide
/// as their widest text plus 3 px ("Macenko" and "Vahadane" ran together
/// in Hindi), and the panel grows with them too.
#[test]
fn hud_adjustment_labels_fit_their_column() {
    use slint_interpreter::Value;
    crate::test_support::run_on_slint_ui_test_thread(|| {
        let ui = window();
        let _english = English(&ui);
        let measure = TextMeasure::new();
        // The Units and Stain Norm labels' texts, translated as the window
        // translates them (assistive technology does not see those labels).
        let mut compiler = slint_interpreter::Compiler::default();
        compiler.set_translation_domain("eov".into());
        compiler.set_bundled_translations_path(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("translations"),
        );
        let source = r#"
            export component Texts {
                out property <string> units: @tr("HudSettingsPanel" => "Units");
                out property <string> stain: @tr("HudSettingsPanel" => "Stain Norm");
                out property <string> mm: @tr("HudSettingsPanel" => "mm");
                out property <string> inches: @tr("HudSettingsPanel" => "in");
                out property <string> none: @tr("Stain normalization" => "None");
                out property <string> macenko: @tr("HudSettingsPanel" => "Macenko");
                out property <string> vahadane: @tr("HudSettingsPanel" => "Vahadane");
            }
        "#;
        let result = spin_on(compiler.build_from_source(source.into(), "texts.slint".into()));
        let texts = result
            .component("Texts")
            .expect("compiles")
            .create()
            .unwrap();
        let text = |name: &str| match texts.get_property(name).unwrap() {
            Value::String(text) => text.to_string(),
            other => panic!("{name}: {other:?}"),
        };
        fill(&ui);
        settle(&ui);
        let mut wider_choices = 0;
        for language in eov_plugin_api::i18n::LANGUAGES {
            crate::i18n::switch_to(&ui, language);
            settle(&ui);
            let tag = language.tag;
            let size = ui.global::<crate::Locale>().invoke_text_size(11.0);
            let choice_size = ui.global::<crate::Locale>().invoke_text_size(10.0);
            // The sliders' labels (each slider is named after its label),
            // and the Units and Stain Norm labels.
            let mut sliders: Vec<_> = ElementHandle::find_by_element_type_name(&ui, "HudSlider")
                .map(|slider| (bounds(&slider).1, slider.accessible_label().unwrap()))
                .collect();
            sliders.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut slider_labels: Vec<_> = elements(&ui, "HudSlider::label-text")
                .iter()
                .map(bounds)
                .collect();
            slider_labels.sort_by(|a, b| a.1.total_cmp(&b.1));
            assert_eq!((sliders.len(), slider_labels.len()), (4, 4), "{tag}");
            let rows = [
                (
                    bounds(&elements(&ui, "HudSettingsPanel::units-label")[0]),
                    text("units"),
                ),
                (
                    bounds(&elements(&ui, "HudSettingsPanel::stain-label")[0]),
                    text("stain"),
                ),
            ];
            let column = sliders
                .iter()
                .map(|(_, label)| measure.width(label, size) + 8.0)
                .chain(rows.iter().map(|(_, label)| measure.width(label, size)))
                .fold(68f32, f32::max)
                .min(150.0);
            // The choices and their texts.
            let choices = [
                ("um", "µm".to_string()),
                ("mm", text("mm")),
                ("in", text("inches")),
                ("none", text("none")),
                ("macenko", text("macenko")),
                ("vahadane", text("vahadane")),
            ]
            .map(|(id, label)| {
                let at = bounds(&elements(&ui, &format!("HudSettingsPanel::{id}-choice"))[0]);
                (label, at)
            });
            let choice = choices
                .iter()
                .map(|(label, _)| measure.width(label, choice_size) + 3.0)
                .fold(160.0 / 3.0, f32::max);
            if choice > 160.0 / 3.0 + 0.5 {
                wider_choices += 1;
            }
            for (label, at) in &choices {
                assert!(
                    same(at.2 - at.0, choice),
                    "{tag}: the choice {label:?} is {at:?}, not {choice} px wide"
                );
            }
            let panel = ElementHandle::find_by_element_type_name(&ui, "HudSettingsPanel")
                .next()
                .unwrap();
            let panel = bounds(&panel);
            assert!(
                same(panel.2 - panel.0, column + 3.0 * choice + 32.0),
                "{tag}: the panel is {panel:?}, its labels need a column of {column} px \
                 and its choices {choice} px each"
            );
            let labels = sliders
                .iter()
                .map(|(_, label)| label.to_string())
                .zip(slider_labels.iter().map(|at| (*at, column - 8.0)))
                .chain(
                    rows.iter()
                        .map(|(at, label)| (label.clone(), (*at, column))),
                );
            for (label, (at, width)) in labels {
                assert!(
                    same(at.2 - at.0, width) && same(at.0, panel.0 + 12.0),
                    "{tag}: {label:?} at {at:?}, not {width} px from {}",
                    panel.0 + 12.0
                );
            }
            // A slider's handle at 0 starts where its touch area does: 7 px
            // after its label. The first Units and Stain Norm choices start
            // 4 px after their labels.
            let mut touches: Vec<_> = elements(&ui, "HudSlider::slider-touch")
                .iter()
                .map(bounds)
                .collect();
            touches.sort_by(|a, b| a.1.total_cmp(&b.1));
            for ((_, label), (at, touch)) in sliders.iter().zip(slider_labels.iter().zip(&touches))
            {
                assert!(
                    same(touch.0 - at.2, 7.0),
                    "{tag}: {label:?} ends at {}, its handle at 0 starts at {}",
                    at.2,
                    touch.0
                );
            }
            for ((at, label), (_, first)) in rows.iter().zip([&choices[0], &choices[3]]) {
                assert!(
                    same(first.0 - at.2, 4.0),
                    "{tag}: {label:?} ends at {}, its first choice starts at {}",
                    at.2,
                    first.0
                );
            }
        }
        assert!(
            wider_choices > 0,
            "no choice needed more room: wider choices went untested"
        );
        ui.hide().unwrap();
    });
}

/// The Save a Copy and Dataset export dialogs, each with the elements that
/// stay in view below its settings: its buttons, and its error.
type ShortDialog = (SetVisible, &'static str, &'static [&'static str]);

const SHORT_DIALOGS: [ShortDialog; 2] = [
    (
        AppWindow::set_save_copy_dialog_visible,
        "SaveCopyDialog::card",
        &["DialogButton"],
    ),
    (
        AppWindow::set_dataset_export_dialog_visible,
        "DatasetExportDialog::card",
        &[
            "DatasetExportDialog::cancel-btn-touch",
            "DatasetExportDialog::confirm-btn-touch",
        ],
    ),
];

/// In a window too short for them (800 × 600 px, the smallest), the Save a
/// Copy and Dataset export dialogs keep their title, their error and their
/// buttons in view and scroll the settings between them, in every language
/// (the buttons were cut off); where the settings fit, nothing scrolls.
#[test]
fn short_windows_scroll_the_dialogs_settings() {
    crate::test_support::run_on_slint_ui_test_thread(|| {
        let ui = AppWindow::new().unwrap();
        ui.window().set_size(slint::PhysicalSize::new(800, 600));
        ui.show().unwrap();
        let _english = English(&ui);
        let error = "The copy cannot be written: ".repeat(6);
        let mut copy = ui.get_save_copy_settings();
        copy.title = "C3L-00004-21.svs".into();
        copy.whole_text = "25895 × 23396 px".into();
        copy.summary = "25895 × 23396 px in 7 levels of 512-px tiles at 0.4942 µm/px.".into();
        copy.error = error.as_str().into();
        ui.set_save_copy_settings(copy);
        let mut dataset = ui.get_dataset_export_settings();
        dataset.error = error.as_str().into();
        ui.set_dataset_export_settings(dataset);
        let window = (0.0, 0.0, 800.0, 600.0);
        // Scrolls a long way down (with the wheel) and lets it settle.
        let scroll_down = |area: &ElementHandle| {
            area.scroll(0.0, -5000.0);
            i_slint_backend_testing::mock_elapsed_time(std::time::Duration::from_secs(2));
            settle(&ui);
        };
        for language in eov_plugin_api::i18n::LANGUAGES {
            crate::i18n::switch_to(&ui, language);
            settle(&ui);
            let tag = language.tag;
            for (set_visible, card, below) in SHORT_DIALOGS {
                set_visible(&ui, true);
                settle(&ui);
                let card = bounds(&elements(&ui, card)[0]);
                assert!(inside(card, window), "{tag}: {card:?}");
                let area = ElementHandle::find_by_element_type_name(&ui, "ScrollArea")
                    .next()
                    .expect("the settings scroll");
                let (top, content) = (bounds(&area), elements(&ui, "ScrollArea::content"));
                let content = bounds(&content[0]);
                assert!(inside(top, card), "{tag}: {top:?} {card:?}");
                assert!(same(content.1, top.1), "{tag}: {content:?} {top:?}");
                assert!(
                    content.3 > top.3 + 1.0,
                    "{tag}: {content:?} fits in {top:?}"
                );
                // The error and the buttons, in view below the settings.
                let mut kept: Vec<_> = below
                    .iter()
                    .flat_map(|name| {
                        ElementHandle::find_by_element_type_name(&ui, name)
                            .chain(ElementHandle::find_by_element_id(&ui, name))
                    })
                    .map(|element| bounds(&element))
                    .collect();
                assert_eq!(kept.len(), 2, "{tag}: {kept:?}");
                let shown_error = find(&ui, AccessibleRole::Text, &error);
                kept.push(bounds(&shown_error));
                for element in kept {
                    assert!(
                        inside(element, card) && element.1 >= top.3,
                        "{tag}: {element:?} is not in view below {top:?} in {card:?}"
                    );
                }
                // Scrolled to the end, the last checkbox is in view.
                scroll_down(&area);
                let scrolled = bounds(&elements(&ui, "ScrollArea::content")[0]);
                assert!(same(scrolled.3, top.3), "{tag}: {scrolled:?} {top:?}");
                assert!(same(scrolled.3 - scrolled.1, content.3 - content.1));
                let last = ElementHandle::find_by_element_type_name(&ui, "CheckRow")
                    .map(|row| bounds(&row))
                    .max_by(|a, b| a.1.total_cmp(&b.1))
                    .unwrap();
                assert!(inside(last, top), "{tag}: {last:?} {top:?}");
                set_visible(&ui, false);
                settle(&ui);
            }
        }
        // Where the settings fit, they do not scroll.
        crate::i18n::switch_to(&ui, eov_plugin_api::i18n::source());
        ui.window().set_size(slint::PhysicalSize::new(1280, 2400));
        settle(&ui);
        for (set_visible, _, _) in SHORT_DIALOGS {
            set_visible(&ui, true);
            settle(&ui);
            let area = ElementHandle::find_by_element_type_name(&ui, "ScrollArea")
                .next()
                .unwrap();
            let (top, content) = (bounds(&area), elements(&ui, "ScrollArea::content"));
            let content = bounds(&content[0]);
            assert_eq!(
                (content.1, content.3),
                (top.1, top.3),
                "as tall as the settings"
            );
            scroll_down(&area);
            assert_eq!(bounds(&elements(&ui, "ScrollArea::content")[0]), content);
            set_visible(&ui, false);
            settle(&ui);
        }
        ui.hide().unwrap();
    });
}

/// The Keyboard Shortcuts panel wraps a long entry (a plugin's tool with
/// the keys it goes with) instead of cutting it, and keeps its key in
/// view; a short entry keeps its height.
#[test]
fn long_shortcut_entries_wrap() {
    crate::test_support::run_on_slint_ui_test_thread(|| {
        let ui = window();
        let long = "Brush (paint a region, over a selected region to extend it; Alt or the eraser erases; [ and ] change its size) "
            .repeat(3);
        let long = long.trim().to_string();
        let row = |heading: bool, label: &str, keys: &str| crate::ShortcutRow {
            heading,
            label: label.into(),
            id: "".into(),
            keys: keys.into(),
        };
        ui.set_shortcut_rows(model(vec![
            row(true, "Annotations (plugin)", ""),
            row(false, "Rectangle", "Ctrl+Alt+F11"),
            row(false, &long, "Ctrl+Alt+F12"),
        ]));
        ui.set_shortcuts_panel_visible(true);
        settle(&ui);
        let short = find(&ui, AccessibleRole::Text, "Rectangle");
        let entry = find(&ui, AccessibleRole::Text, &long);
        let key = bounds(&find(&ui, AccessibleRole::Text, "Ctrl+Alt+F12"));
        assert!(short.size().height <= 24.0, "{:?}", short.size());
        assert!(
            entry.size().height >= 3.0 * short.size().height,
            "{:?} {:?}",
            entry.size(),
            short.size()
        );
        assert!(bounds(&entry).2 <= key.0, "the entry stops before its key");
        let heading = find(&ui, AccessibleRole::Text, "Annotations (plugin)");
        assert!(
            bounds(&heading).3 <= bounds(&short).1,
            "the heading comes first"
        );
        ui.set_shortcuts_panel_visible(false);
        ui.hide().unwrap();
    });
}
