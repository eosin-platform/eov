//! eov's words for colour processing in the interface's language (CLR-9):
//! the colour badge and warnings of a pane, the export dialog's notes,
//! stain, stain-preset and stain-model names, and stain-fit summaries.
//!
//! `eov_common` describes processing in English, for exported files,
//! burned-in captions, scripts and logs ([`ProcessingSummary::badge`],
//! [`ColorSettings::warnings`], [`StainEstimate::summary`]). It keeps the
//! same facts as data beside those texts, and this module words them for
//! the window. In English both say exactly the same (tested), so a label of
//! processed or synthetic pixels never says less in another language.
//!
//! [`ColorSettings::warnings`]: eov_common::processing::ColorSettings::warnings

use eov_common::processing::{
    ColorWarning, NotApplied, ProcessingSummary, StainAdjustment, StainModel,
};
use eov_common::stain::{
    EstimateSource, FitFallback, StainEstimate, StainModelMismatch, StainPreset,
    StainTransformError,
};
use eov_plugin_api::i18n::{format_decimal, isolate, tr};

/// `items` as a list in the interface's language ("a, b, c").
pub(crate) fn list(items: impl IntoIterator<Item = String>) -> String {
    // Translators: what separates the items of a list ("Hematoxylin ×1.20,
    // Eosin hidden"): a comma and a space in English.
    let separator = tr!("List separator" => ", ");
    items.into_iter().collect::<Vec<_>>().join(&separator)
}

// ─── Names ───────────────────────────────────────────────────────────────────

/// A stain's name as the window shows it: the stains of eov's presets in
/// the interface's language (abbreviations such as DAB as they are), other
/// names (a custom stain model's) as written.
pub(crate) fn stain_name(name: &str) -> String {
    known_stain_name(name).unwrap_or_else(|| name.to_string())
}

/// The name of a stain of eov's presets ([`StainPreset`]), if `name` is one.
fn known_stain_name(name: &str) -> Option<String> {
    Some(match name {
        // Translators: a histological stain (dye), as in "Isolated stain:
        // Hematoxylin".
        "Hematoxylin" => tr!("Stain" => "Hematoxylin"),
        // Translators: a histological stain (dye).
        "Eosin" => tr!("Stain" => "Eosin"),
        // Translators: a histological stain (dye) of Masson's trichrome.
        "Methyl blue" => tr!("Stain" => "Methyl blue"),
        // Translators: a histological stain (dye) of Masson's trichrome.
        "Ponceau-fuchsin" => tr!("Stain" => "Ponceau-fuchsin"),
        // Abbreviations (diaminobenzidine, aminoethylcarbazole, periodic
        // acid–Schiff), written so in every language.
        "DAB" | "AEC" | "PAS" => name.to_string(),
        _ => return None,
    })
}

/// The name of a special stain a slide's label names (one of
/// [`eov_common::stain::special_stain_names`]) in the interface's language.
pub(crate) fn special_stain_name(name: &str) -> String {
    known_special_stain_name(name).unwrap_or_else(|| name.to_string())
}

fn known_special_stain_name(name: &str) -> Option<String> {
    Some(match name {
        // Translators: a staining method of histology, named after a person.
        "Masson trichrome" => tr!("Stain" => "Masson trichrome"),
        // Translators: a staining method of histology, named after a person.
        "Giemsa" => tr!("Stain" => "Giemsa"),
        // Translators: a staining method of histology (reticulin fibres).
        "Reticulin" => tr!("Stain" => "Reticulin"),
        // Translators: a staining method of histology (for amyloid).
        "Congo red" => tr!("Stain" => "Congo red"),
        // Translators: a staining method of histology, named after a person.
        "Van Gieson" => tr!("Stain" => "Van Gieson"),
        // Translators: a staining method of histology for elastic fibres (such
        // as Verhoeff or orcein): the name of the stain, not of the protein.
        "Elastic" => tr!("Stain" => "Elastic"),
        // Translators: a staining method of histology (for mucins).
        "Alcian blue" => tr!("Stain" => "Alcian blue"),
        // Translators: a staining method of histology (for iron), named
        // after a person.
        "Perls' Prussian blue" => tr!("Stain" => "Perls' Prussian blue"),
        // Translators: a staining method of microbiology, named after a
        // person.
        "Gram" => tr!("Stain" => "Gram"),
        // Translators: a staining method of histology (for fungi), named
        // after a person.
        "Grocott methenamine silver" => tr!("Stain" => "Grocott methenamine silver"),
        // Translators: a staining method of histology (for mucins).
        "Mucicarmine" => tr!("Stain" => "Mucicarmine"),
        // Translators: a staining method of microbiology (acid-fast
        // bacteria), named after two people.
        "Ziehl–Neelsen" => tr!("Stain" => "Ziehl–Neelsen"),
        // Periodic acid–Schiff, an abbreviation in every language.
        "PAS" => name.to_string(),
        _ => return None,
    })
}

/// A stain preset's name as the window shows it ([`StainPreset::label`]
/// in the interface's language; abbreviations such as H&E and H-DAB as
/// they are).
pub(crate) fn preset_label(preset: StainPreset) -> String {
    match preset {
        // Translators: the stain vectors of hematoxylin and eosin that
        // Macenko et al. published as a reference.
        StainPreset::HematoxylinEosinMacenko => tr!("Stain model" => "H&E (Macenko reference)"),
        // Translators: a staining method of histology, named after a person.
        StainPreset::MassonTrichrome => tr!("Stain" => "Masson trichrome"),
        StainPreset::HematoxylinEosin
        | StainPreset::HematoxylinDab
        | StainPreset::HematoxylinEosinDab
        | StainPreset::HematoxylinPas
        | StainPreset::HematoxylinAec => preset.label().to_string(),
    }
}

/// A stain model's name as the window shows it ([`StainModel::label`]).
pub(crate) fn stain_model_label(model: &StainModel) -> String {
    match model {
        StainModel::Preset(preset) => preset_label(*preset),
        // Translators: a stain model of the user's own stain vectors; {}
        // lists its stains.
        StainModel::Custom(basis) => tr!(
            "custom ({})",
            list(basis.names().into_iter().map(stain_name))
        ),
    }
}

// ─── The colour badge, warnings and notes ────────────────────────────────────

/// The colour badge of a pane or an export ([`ProcessingSummary::badge`])
/// in the interface's language: every processing that changed the pixels,
/// and what was asked for but not applied. `None` when nothing was.
pub(crate) fn badge(summary: &ProcessingSummary) -> Option<String> {
    let mut parts = Vec::new();
    // The method's name (Macenko, Vahadane) is the same in every language.
    if let Some(method) = &summary.normalization {
        parts.push(if summary.normalization_pending {
            // Translators: on a pane while the slide's stains are still being
            // fitted: the colours shown are NOT normalised yet (waiting, not
            // partly applied); {} is the method (Macenko or Vahadane).
            tr!("Colour normalisation pending ({})", method)
        } else if summary.normalization_fallback.is_some() {
            // Translators: the slide's own stains could not be fitted, so
            // generic H&E stain vectors were used instead (a fallback); {} is
            // the method (Macenko or Vahadane).
            tr!("Colour-normalised ({}, fallback stain vectors)", method)
        } else {
            // Translators: {} is the method (Macenko or Vahadane).
            tr!("Colour-normalised ({})", method)
        });
    }
    if let Some(stain) = &summary.isolated {
        // Translators: only this stain is shown, in grey; {} is the stain
        // (Hematoxylin, Eosin, …).
        parts.push(tr!("Isolated stain: {}", stain_name(stain)));
    }
    if !summary.stain_adjustments.is_empty() {
        // Translators: the stains' intensities were changed; {} lists the
        // changes ("Hematoxylin ×1.20, Eosin hidden").
        parts.push(tr!("Stain-adjusted ({})", stain_adjustments(summary)));
    }
    if !summary.tone.is_empty() {
        // Translators: a badge, also written on exported images: the pixel
        // tones were changed with gamma, brightness or contrast (not the
        // view's position or zoom).
        parts.push(tr!("Display-adjusted"));
    }
    if summary.sharpness.is_some() {
        // Translators: the slide is shown sharpened.
        parts.push(tr!("Sharpened"));
    }
    if !summary.not_applied.is_empty() {
        // Translators: some of the colour processing asked for could not
        // be done (the warnings say what and why).
        parts.push(tr!("Some adjustments not applied"));
    }
    (!parts.is_empty()).then(|| parts.join(" · "))
}

/// The summary's stain adjustments as a list. A summary without its data
/// (read back from a file) lists them as it wrote them.
fn stain_adjustments(summary: &ProcessingSummary) -> String {
    let details = &summary.stain_adjustment_details;
    if details.len() == summary.stain_adjustments.len() {
        list(details.iter().map(stain_adjustment))
    } else {
        list(summary.stain_adjustments.iter().cloned())
    }
}

/// One stain's adjustment ("Eosin hidden", "Hematoxylin ×1.20").
pub(crate) fn stain_adjustment(adjustment: &StainAdjustment) -> String {
    match adjustment {
        // Translators: {} is a stain (Eosin, …) that is not shown.
        StainAdjustment::Hidden { stain } => tr!("Stain" => "{} hidden", stain_name(stain)),
        // Translators: a stain shown at a multiple of its intensity
        // ("Hematoxylin ×1.20").
        StainAdjustment::Scaled { stain, factor } => tr!(
            "{stain} ×{factor}",
            stain = stain_name(stain),
            factor = format_decimal(f64::from(*factor), 2)
        ),
    }
}

/// Why requested processing is not applied ("isolating stain 3 (the stain
/// model has 2 stains)"): an entry of [`ProcessingSummary::not_applied`].
pub(crate) fn not_applied(item: &NotApplied) -> String {
    match item {
        // Translators: {method} is Macenko or Vahadane; {reason} says why
        // the normalisation cannot be applied.
        NotApplied::Normalization { method, error } => tr!(
            "{method} stain normalisation ({reason})",
            method = isolate(method.label()),
            reason = transform_error(error)
        ),
        // Translators: {stain} is the number of a stain, {n} how many
        // stains the stain model has.
        NotApplied::Isolation { index, stains } => tr!(
            "isolating stain {stain} (the stain model has {n} stain)"
                | "isolating stain {stain} (the stain model has {n} stains)" % *stains,
            stain = index + 1
        ),
        NotApplied::ExtraStains { stains } => {
            // Translators: an item of "Not applied: …": the stains of some
            // adjustments are not in the stain model; {n} is how many stains
            // the stain model has.
            tr!(
                "adjusting stains the stain model does not have (it has {n} stain)"
                    | "adjusting stains the stain model does not have (it has {n} stains)"
                        % *stains
            )
        }
        NotApplied::StainStage(error) => transform_error(error),
    }
}

fn transform_error(error: &StainTransformError) -> String {
    match error {
        StainTransformError::StainCountMismatch {
            source_stains,
            target,
        } => {
            // Translators: the stain normalisation's target has another
            // number of stains ({n}) than the slide's fit ("the source").
            tr!(
                "the normalisation target has {n} stain but the source has {source}"
                    | "the normalisation target has {n} stains but the source has {source}"
                        % *target,
                source = source_stains
            )
        }
        // Translators: {index} is the number of a stain, {n} how many
        // stains the stain model ("basis") has.
        StainTransformError::NoSuchStain { index, stains } => tr!(
            "isolated stain {index} does not exist (the basis has {n} stain)"
                | "isolated stain {index} does not exist (the basis has {n} stains)" % *stains,
            index = index
        ),
    }
}

/// [`not_applied`] for each entry of the summary's list. A summary without
/// its data (read back from a file) gives them as it wrote them.
pub(crate) fn not_applied_reasons(summary: &ProcessingSummary) -> Vec<String> {
    if summary.not_applied_details.len() == summary.not_applied.len() {
        summary
            .not_applied_details
            .iter()
            .map(not_applied)
            .collect()
    } else {
        summary.not_applied.clone()
    }
}

/// A problem with a pane's colour settings
/// ([`ColorSettings::warning_list`]) in the interface's language.
///
/// [`ColorSettings::warning_list`]: eov_common::processing::ColorSettings::warning_list
pub(crate) fn warning(warning: &ColorWarning) -> String {
    match warning {
        // Translators: {} is a stain model (H-DAB, Masson trichrome, …).
        ColorWarning::NormalizationNeedsHe(model) => tr!(
            "Stain normalisation only supports H&E; it is disabled for the {} stain model.",
            stain_model_label(model)
        ),
        // Translators: {} is the stain the slide's file names ("Ki-67",
        // "IHC").
        ColorWarning::StainLabel(StainModelMismatch::Immunohistochemistry { label }) => tr!(
            "This slide is labelled \"{}\" (immunohistochemistry), but the H&E stain model is selected; DAB and other chromogens will be shown in the wrong colours. Use the H-DAB stain preset instead.",
            isolate(label)
        ),
        // Translators: {} is a staining method (Giemsa, Congo red, …).
        ColorWarning::StainLabel(StainModelMismatch::Special(name)) => tr!(
            "This slide is labelled as a {} stain, but the H&E stain model is selected; its colours cannot be represented by hematoxylin and eosin.",
            special_stain_name(name)
        ),
        // Translators: {} says what was asked for and why it is not done
        // ("isolating stain 3 (the stain model has 2 stains)").
        ColorWarning::NotApplied(item) => tr!("Not applied: {}.", not_applied(item)),
    }
}

// ─── Stain fits ──────────────────────────────────────────────────────────────

/// A stain fit in one line ([`StainEstimate::summary`]) in the interface's
/// language.
pub(crate) fn estimate_summary(estimate: &StainEstimate) -> String {
    let source = estimate_source(&estimate.source);
    // The method's name (Macenko, Vahadane) is the same in every language;
    // isolated, so that in a right-to-left language the Latin name at the
    // start does not make the whole line run left to right.
    let method = isolate(estimate.method.label());
    match &estimate.fallback {
        // Translators: a stain fit: the method (Macenko or Vahadane), the
        // pixels it used, how much of them was tissue, and the angle
        // between the fitted stains.
        None => tr!(
            "{method} fit ({source}; {percent}% tissue; stains {angle}° apart)",
            method = method,
            source = source,
            percent = format_decimal(f64::from(estimate.tissue_fraction * 100.0), 1),
            angle = format_decimal(f64::from(estimate.basis.min_angle_deg()), 1)
        ),
        // Translators: a stain fit that was not used: the method (Macenko
        // or Vahadane), why, and the pixels it used. The standard
        // ("reference") stain vectors were used instead.
        Some(reason) => tr!(
            "{method} fit rejected ({reason}); reference stain vectors used ({source})",
            method = method,
            reason = fit_fallback(reason),
            source = source
        ),
    }
}

/// Where a fit's pixels came from.
fn estimate_source(source: &EstimateSource) -> String {
    match source {
        // Translators: a stain fit's pixels came from a plugin or a script.
        EstimateSource::Pixels => tr!("supplied pixels"),
        EstimateSource::Slide {
            level,
            sparse: true,
            ..
        } => {
            // Translators: a stain fit read blocks spread over a pyramid
            // level of the slide; {} is the level.
            tr!("slide, sparse sample of level {}", level)
        }
        // Translators: a stain fit read a pyramid level of the slide; {} is
        // the level.
        EstimateSource::Slide { level, .. } => tr!("slide level {}", level),
        EstimateSource::Region {
            width,
            height,
            level,
            ..
        } => {
            // Translators: a stain fit read a region of the slide: its size
            // in pixels and the pyramid level.
            tr!(
                "{width}×{height} px region, level {level}",
                width = width,
                height = height,
                level = level
            )
        }
    }
}

/// Why a stain fit was not used ([`FitFallback::describe`]).
pub(crate) fn fit_fallback(fallback: &FitFallback) -> String {
    match fallback {
        FitFallback::InsufficientTissue {
            tissue_pixels,
            tissue_fraction,
        } => {
            // Translators: too little of the slide was tissue to fit its
            // stains: how many pixels, and which share of the slide.
            tr!(
                "insufficient tissue ({pixels} px, {percent}%)",
                pixels = tissue_pixels,
                percent = format_decimal(f64::from(*tissue_fraction * 100.0), 2)
            )
        }
        FitFallback::IllConditioned {
            min_angle_deg,
            condition_number,
        } => {
            // Translators: the fitted stains were too alike to tell apart:
            // the angle between them, and the fit's condition number.
            tr!(
                "ill-conditioned fit (stain vectors {angle}° apart, condition {condition})",
                angle = format_decimal(f64::from(*min_angle_deg), 1),
                condition = format_decimal(f64::from(*condition_number), 1)
            )
        }
        // Translators: the fitted stains were too far from those of
        // hematoxylin and eosin.
        FitFallback::Implausible { deviation_deg } => tr!(
            "implausible stain vectors ({}° from the reference)",
            format_decimal(f64::from(*deviation_deg), 1)
        ),
        // Translators: a stain fit failed.
        FitFallback::FitFailed => tr!("the fit did not converge"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eov_common::postprocess::ToneAdjustments;
    use eov_common::processing::ColorSettings;
    use eov_common::stain::{NormalizationTarget, StainChannel, StainFitMethod};
    use eov_plugin_api::i18n::{self, Catalog};

    fn estimate(method: StainFitMethod) -> StainEstimate {
        StainEstimate {
            format_version: 1,
            method,
            basis: StainPreset::HematoxylinEosin.basis(),
            max_concentrations: vec![1.6, 0.9],
            i0: [242.0, 240.0, 244.0],
            background_found: true,
            examined_pixels: 100,
            tissue_pixels: 50,
            tissue_fraction: 0.5,
            samples: 50,
            min_angle_deg: 35.0,
            condition_number: 3.2,
            fallback: None,
            source: EstimateSource::Slide {
                level: 2,
                downsample: 16.0,
                sparse: false,
            },
            slide_identity: None,
        }
    }

    fn fallbacks() -> [FitFallback; 4] {
        [
            FitFallback::FitFailed,
            FitFallback::InsufficientTissue {
                tissue_pixels: 12,
                tissue_fraction: 0.000_123,
            },
            FitFallback::IllConditioned {
                min_angle_deg: 4.25,
                condition_number: f32::INFINITY,
            },
            FitFallback::Implausible {
                deviation_deg: 41.0,
            },
        ]
    }

    /// Settings that meet every part of the badge and every warning.
    fn settings_variants() -> Vec<ColorSettings> {
        let tone = ToneAdjustments::new(1.3, 0.04, 1.2);
        let legacy = |normalization, h, h_visible, e, e_visible, isolated, sharpness| {
            ColorSettings::from_legacy(
                normalization,
                h,
                h_visible,
                e,
                e_visible,
                isolated,
                tone,
                sharpness,
            )
        };
        use eov_common::StainNormalization::{Macenko, None as Off, Vahadane};
        let mut variants = vec![
            ColorSettings::default(),
            legacy(Off, 1.5, true, 0.7, true, 0, 0.0),
            legacy(Macenko, 1.0, true, 1.0, true, 0, 0.4),
            legacy(Vahadane, 1.2, true, 1.0, false, 0, 0.0),
            legacy(Off, 2.0, true, 1.0, true, 1, 0.0),
            legacy(Off, 1.0, true, 1.0, true, 2, 0.2),
            ColorSettings {
                stain_model: StainModel::Preset(StainPreset::HematoxylinEosinDab),
                channels: vec![
                    StainChannel::default(),
                    StainChannel::default(),
                    StainChannel {
                        intensity: 1.8,
                        visible: true,
                    },
                ],
                ..ColorSettings::default()
            },
            ColorSettings {
                stain_model: StainModel::Preset(StainPreset::HematoxylinDab),
                channels: vec![StainChannel {
                    intensity: 1.5,
                    visible: true,
                }],
                isolated: Some(2),
                ..ColorSettings::default()
            },
            ColorSettings {
                isolated: Some(4),
                ..ColorSettings::default()
            },
            ColorSettings {
                channels: vec![
                    StainChannel::default(),
                    StainChannel {
                        intensity: 0.0,
                        visible: false,
                    },
                    StainChannel {
                        intensity: 2.0,
                        visible: true,
                    },
                ],
                isolated: Some(0),
                tone: ToneAdjustments::new(0.8, -0.05, 1.25),
                ..ColorSettings::default()
            },
        ];
        for preset in [StainPreset::MassonTrichrome, StainPreset::HematoxylinPas] {
            variants.push(ColorSettings {
                normalization: Some(StainFitMethod::Macenko),
                stain_model: StainModel::Preset(preset),
                channels: vec![StainChannel {
                    intensity: 0.5,
                    visible: true,
                }],
                ..ColorSettings::default()
            });
        }
        variants.push(ColorSettings {
            normalization: Some(StainFitMethod::Vahadane),
            stain_model: StainModel::Custom(
                eov_common::stain::StainBasis::new(vec![
                    eov_common::stain::Stain::new("Nuclear", [0.6, 0.7, 0.3]),
                    eov_common::stain::Stain::new("Eosin", [0.1, 0.9, 0.3]),
                ])
                .unwrap(),
            ),
            isolated: Some(1),
            ..ColorSettings::default()
        });
        let mut three_stains = legacy(Macenko, 1.3, true, 1.0, true, 0, 0.0);
        three_stains.target = Some(NormalizationTarget {
            basis: StainPreset::HematoxylinEosinDab.basis(),
            max_concentrations: vec![1.0; 3],
        });
        variants.push(three_stains);
        variants
    }

    /// In English the window says what `eov_common` says, word for word:
    /// nothing a label of processed pixels says is lost.
    #[test]
    fn english_texts_are_common_s() {
        let mut estimates = vec![
            None,
            Some(estimate(StainFitMethod::Macenko)),
            Some(estimate(StainFitMethod::Vahadane)),
        ];
        for fallback in fallbacks() {
            let mut fell_back = estimate(StainFitMethod::Macenko);
            fell_back.fallback = Some(fallback);
            estimates.push(Some(fell_back));
        }
        let mut checked = 0;
        for settings in settings_variants() {
            for estimate in &estimates {
                let pipeline = settings.pipeline(estimate.as_ref());
                let summary = pipeline.summary();
                assert_eq!(badge(summary), summary.badge(), "{settings:?}");
                assert_eq!(not_applied_reasons(summary), summary.not_applied);
                checked += usize::from(summary.badge().is_some());
            }
            for label in [
                None,
                Some("IHC"),
                Some(" Ki-67 "),
                Some("Giemsa"),
                Some("PAS"),
            ] {
                let words: Vec<String> = settings.warning_list(label).iter().map(warning).collect();
                assert_eq!(words, settings.warnings(label), "{settings:?} {label:?}");
            }
        }
        assert!(checked > 40, "{checked}");
        for estimate in estimates.iter().flatten() {
            assert_eq!(estimate_summary(estimate), estimate.summary());
            if let Some(fallback) = &estimate.fallback {
                assert_eq!(fit_fallback(fallback), fallback.describe());
            }
        }
        let mut sources = estimate(StainFitMethod::Vahadane);
        for source in [
            EstimateSource::Pixels,
            EstimateSource::Slide {
                level: 3,
                downsample: 8.0,
                sparse: true,
            },
            EstimateSource::Region {
                x: 5,
                y: 6,
                width: 640,
                height: 480,
                level: 1,
                downsample: 2.0,
            },
        ] {
            sources.source = source;
            assert_eq!(estimate_summary(&sources), sources.summary());
        }
        // A summary read back from a file has no data: its texts are shown.
        let summary = settings_variants()[9].pipeline(None).summary().clone();
        let json = serde_json::to_string(&summary).unwrap();
        let read: ProcessingSummary = serde_json::from_str(&json).unwrap();
        assert!(read.stain_adjustment_details.is_empty());
        assert_eq!(badge(&read), summary.badge());
        assert_eq!(not_applied_reasons(&read), summary.not_applied);
    }

    /// Every stain of the presets and every special stain a label can name
    /// is known: a new one cannot go untranslated unnoticed.
    #[test]
    fn every_stain_eov_names_is_known() {
        for preset in StainPreset::ALL {
            for name in preset.basis().names() {
                assert!(known_stain_name(name).is_some(), "{preset:?}: {name}");
            }
            assert!(!preset_label(preset).is_empty());
        }
        for name in eov_common::stain::special_stain_names() {
            assert!(known_special_stain_name(name).is_some(), "{name}");
        }
        assert_eq!(stain_name("Nuclear"), "Nuclear", "custom names as written");
        assert_eq!(preset_label(StainPreset::HematoxylinDab), "H-DAB");
    }

    /// English again however a test ends.
    struct English;

    impl Drop for English {
        fn drop(&mut self) {
            i18n::set_catalog(i18n::source(), None);
        }
    }

    /// In another language the words, the stain names and the decimal
    /// separator are the language's.
    #[test]
    fn texts_follow_the_interface_s_language() {
        let german = i18n::by_tag("de").unwrap();
        let messages: [(Option<&str>, &str, &[&str]); 6] = [
            (Some("Stain"), "Hematoxylin", &["Hämatoxylin"]),
            (Some("Stain"), "Eosin", &["Eosin"]),
            (Some("Stain"), "{} hidden", &["{} ausgeblendet"]),
            (None, "Stain-adjusted ({})", &["Färbungen angepasst ({})"]),
            (None, "Colour-normalised ({})", &["Farbnormalisiert ({})"]),
            (None, "Isolated stain: {}", &["Isolierte Färbung: {}"]),
        ];
        let catalog = Catalog::from_messages(german, "", messages).unwrap();
        let _english = English;
        i18n::set_catalog(german, Some(catalog));
        let settings = ColorSettings::from_legacy(
            eov_common::StainNormalization::Macenko,
            1.2,
            true,
            1.0,
            false,
            1,
            ToneAdjustments::default(),
            0.0,
        );
        let pipeline = settings.pipeline(Some(&estimate(StainFitMethod::Macenko)));
        assert_eq!(
            badge(pipeline.summary()).as_deref(),
            Some(
                "Farbnormalisiert (Macenko) · Isolierte Färbung: Hämatoxylin · Färbungen \
                 angepasst (Hämatoxylin ×1,20, Eosin ausgeblendet)"
            )
        );
        // The English stays as it was, for files and scripts.
        assert!(
            pipeline
                .summary()
                .badge()
                .unwrap()
                .starts_with("Colour-normalised (Macenko) · Isolated stain: Hematoxylin")
        );
        let mut fell_back = estimate(StainFitMethod::Macenko);
        fell_back.fallback = Some(FitFallback::Implausible {
            deviation_deg: 41.5,
        });
        assert!(
            estimate_summary(&fell_back).contains("(41,5° from the reference)"),
            "{}",
            estimate_summary(&fell_back)
        );
    }
}
