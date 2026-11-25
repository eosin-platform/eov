//! The stain normalisation target from the viewport menu (AUDIT §14.5,
//! CLR-1): use this slide or its region of interest as the appearance every
//! normalised pane is mapped onto, save the target to share it, load one,
//! or go back to Macenko's reference ([`crate::stain_target`]).

use super::{NotifyLevel, UiHost, with_host};
use crate::color::SlideFit;
use crate::stain_target::{self, StainTarget};
use crate::state::PaneId;
use eov_common::WsiFile;
use eov_common::stain::{Level0Rect, StainEstimateOptions, StainFitMethod, estimate_region_stain};
use eov_plugin_api::i18n::{isolate, tr};

impl UiHost {
    /// The viewport menu's `stain-target-…` commands for the menu opened on
    /// `pane`. Returns whether it handled `command`.
    pub(super) fn stain_target_command(&self, command: &str, file_id: i32, pane: usize) -> bool {
        match command {
            "stain-target-slide" => self.stain_target_from_slide(file_id, pane),
            "stain-target-region" => self.stain_target_from_region(file_id, pane),
            "stain-target-save" => self.save_stain_target(),
            "stain-target-load" => self.load_stain_target(),
            "stain-target-reset" => {
                self.use_stain_target(None);
            }
            _ => return false,
        }
        true
    }

    /// The slide, its name and the normalisation method the pane uses.
    fn normalising(&self, file_id: i32, pane: usize) -> Option<(WsiFile, String, StainFitMethod)> {
        let state = self.state.read();
        let file = state.get_file(file_id)?;
        let method = StainFitMethod::from_normalization(
            file.pane_state(PaneId(pane))?.hud.stain_normalization,
        )?;
        Some((file.wsi.clone(), file.filename.clone(), method))
    }

    fn stain_target_from_slide(&self, file_id: i32, pane: usize) {
        let Some((wsi, name, method)) = self.normalising(file_id, pane) else {
            return;
        };
        match crate::color::slide_fit(file_id, &wsi, method) {
            SlideFit::Ready(estimate) if estimate.is_fallback() => self.notify(
                NotifyLevel::Warning,
                &tr!("This slide cannot be the stain target"),
                Some(&crate::processing_text::estimate_summary(&estimate)),
                None,
            ),
            SlideFit::Ready(estimate) => {
                self.use_stain_target(Some(StainTarget::new(name, (*estimate).clone())))
            }
            SlideFit::Pending => self.status(&tr!(
                "The slide's stains are still being fitted; try again in a moment"
            )),
            SlideFit::Failed(reason) => self.notify(
                NotifyLevel::Warning,
                &tr!("This slide cannot be the stain target"),
                Some(&reason),
                None,
            ),
        }
    }

    fn stain_target_from_region(&self, file_id: i32, pane: usize) {
        let Some((wsi, name, method)) = self.normalising(file_id, pane) else {
            return;
        };
        let Some(roi) = self
            .state
            .read()
            .get_file(file_id)
            .and_then(|file| file.roi)
        else {
            self.status(&tr!("Draw a region of interest first (R)"));
            return;
        };
        let job = self.start_host_job(&tr!("Fitting the stains of the region"), None);
        let spawned = std::thread::Builder::new()
            .name("eov-stain-target".to_string())
            .spawn({
                let job = job.clone();
                move || {
                    let region = Level0Rect::new(roi.x, roi.y, roi.width, roi.height);
                    let result = estimate_region_stain(
                        &wsi,
                        region,
                        method,
                        &StainEstimateOptions::for_region(),
                    )
                    .map_err(|error| error.to_string());
                    let _ = slint::invoke_from_event_loop(move || {
                        with_host(|host| {
                            host.finish_host_job(&job);
                            match result {
                                Ok(estimate) if estimate.is_fallback() => host.notify(
                                    NotifyLevel::Warning,
                                    &tr!("This region cannot be the stain target"),
                                    Some(&crate::processing_text::estimate_summary(&estimate)),
                                    None,
                                ),
                                Ok(estimate) => host.use_stain_target(Some(StainTarget::new(
                                    // Translators: the name of a stain target
                                    // taken from a slide's region of interest.
                                    tr!("{} (region)", name),
                                    estimate,
                                ))),
                                Err(error) => host.notify(
                                    NotifyLevel::Error,
                                    &tr!("Could not fit the region's stains"),
                                    Some(&error),
                                    None,
                                ),
                            }
                        });
                    });
                }
            });
        if let Err(error) = spawned {
            self.finish_host_job(&job);
            self.notify(
                NotifyLevel::Error,
                &tr!("Could not fit the region's stains"),
                Some(&error.to_string()),
                None,
            );
        }
    }

    fn save_stain_target(&self) {
        let Some(target) = stain_target::current() else {
            self.status(&tr!(
                "The reference stain target is in use; there is nothing to save"
            ));
            return;
        };
        let Some(path) = rfd::FileDialog::new()
            .set_title(tr!("Save the stain target"))
            .add_filter(tr!("Stain target"), &["json"])
            .set_file_name(format!("{}.json", sanitize(&target.name)))
            .save_file()
        else {
            return;
        };
        match target.save(&path) {
            Ok(()) => self.notify(
                NotifyLevel::Success,
                &tr!("Saved the stain target \"{}\"", isolate(&target.name)),
                None,
                None,
            ),
            Err(error) => self.notify(
                NotifyLevel::Error,
                &tr!("Could not save the stain target"),
                Some(&error),
                None,
            ),
        }
    }

    fn load_stain_target(&self) {
        let Some(path) = rfd::FileDialog::new()
            .set_title(tr!("Load a stain target"))
            .add_filter(tr!("Stain target"), &["json"])
            .pick_file()
        else {
            return;
        };
        match StainTarget::load(&path) {
            Ok(target) if target.estimate.is_fallback() => self.notify(
                NotifyLevel::Warning,
                &tr!("This file cannot be the stain target"),
                Some(&crate::processing_text::estimate_summary(&target.estimate)),
                None,
            ),
            Ok(target) => self.use_stain_target(Some(target)),
            Err(error) => self.notify(
                NotifyLevel::Error,
                &tr!("Could not load the stain target"),
                Some(&error),
                None,
            ),
        }
    }

    /// Makes `target` the target in use and redraws every pane.
    fn use_stain_target(&self, target: Option<StainTarget>) {
        let title = match &target {
            Some(target) => tr!("Normalising to \"{}\"", isolate(&target.name)),
            None => tr!("Normalising to the reference H&E (Macenko)"),
        };
        if let Err(error) = stain_target::set(target) {
            self.notify(
                NotifyLevel::Error,
                &tr!("Could not change the stain target"),
                Some(&error),
                None,
            );
            return;
        }
        {
            let mut state = self.state.write();
            for file in state.open_files.iter_mut() {
                file.invalidate_render_state();
            }
            state.request_render();
        }
        self.request_render();
        self.notify(
            NotifyLevel::Info,
            &title,
            // Translators: "exports follow" means that exported images with
            // stain normalisation on are mapped onto this appearance too; it
            // does not mean that a list of exports comes next.
            Some(&tr!(
                "Every pane with stain normalisation on maps its slide onto this appearance; exports follow."
            )),
            None,
        );
    }
}

/// A file name from a target name.
fn sanitize(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || matches!(c, '-' | '_' | ' ' | '(' | ')') {
                c
            } else {
                '_'
            }
        })
        .collect();
    let cleaned = cleaned.trim();
    if cleaned.is_empty() {
        "stain-target".to_string()
    } else {
        cleaned.to_string()
    }
}
