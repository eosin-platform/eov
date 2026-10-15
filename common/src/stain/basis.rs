//! Stain vectors, stain bases and built-in stain presets.
//!
//! A *stain basis* is the ordered set of unit optical-density (OD) vectors —
//! one per stain — that the Beer–Lambert colour model decomposes a pixel
//! into. Bases with one to three stains are supported; with fewer than three
//! stains the part of a pixel's absorbance that the basis cannot explain is
//! kept as a *residual* by every transform in this crate (see
//! [`StainTransform`](super::StainTransform)).

use serde::{Deserialize, Serialize};

use super::linalg::{self, Vec3};

/// One stain: a display name and its unit optical-density vector `[R, G, B]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Stain {
    /// Human-readable stain name, e.g. `"Hematoxylin"`.
    pub name: String,
    /// Optical density absorbed per unit concentration in the R, G and B
    /// channels (natural-log OD, unit length).
    pub od: [f32; 3],
}

impl Stain {
    pub fn new(name: impl Into<String>, od: [f32; 3]) -> Self {
        Self {
            name: name.into(),
            od,
        }
    }
}

/// Why a set of stain vectors cannot form a [`StainBasis`].
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum StainBasisError {
    #[error("a stain basis needs at least one stain")]
    Empty,
    #[error("a stain basis supports at most 3 stains, got {0}")]
    TooMany(usize),
    #[error("stain {0} has a non-finite component")]
    NonFinite(usize),
    #[error("stain {0} has a negative optical-density component")]
    Negative(usize),
    #[error("stain {0} is a zero vector")]
    Zero(usize),
    #[error("the stain vectors are (nearly) linearly dependent (smallest angle {0:.2}°)")]
    Degenerate(f32),
}

/// An ordered set of 1–3 linearly independent stain vectors.
///
/// Construction normalises every vector to unit length and rejects
/// non-physical input (negative or non-finite components, zero vectors,
/// collinear stains). Deserialisation applies the same validation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "Vec<Stain>", into = "Vec<Stain>")]
pub struct StainBasis {
    stains: Vec<Stain>,
}

/// Smallest angle between two stains that [`StainBasis::new`] accepts.
/// Fit-quality checks (see `FitChecks`) are much stricter; this only guards
/// against numerically singular matrices.
const MIN_BASIS_ANGLE_DEG: f32 = 0.5;

impl StainBasis {
    /// Builds a basis from explicit stain vectors (custom vectors, CLR-8).
    pub fn new(stains: Vec<Stain>) -> Result<Self, StainBasisError> {
        if stains.is_empty() {
            return Err(StainBasisError::Empty);
        }
        if stains.len() > 3 {
            return Err(StainBasisError::TooMany(stains.len()));
        }
        let mut normalized = Vec::with_capacity(stains.len());
        for (index, stain) in stains.into_iter().enumerate() {
            if stain.od.iter().any(|v| !v.is_finite()) {
                return Err(StainBasisError::NonFinite(index));
            }
            if stain.od.iter().any(|&v| v < -1e-4) {
                return Err(StainBasisError::Negative(index));
            }
            let clamped = stain.od.map(|v| v.max(0.0));
            let length = linalg::norm(clamped);
            // Already-normalised vectors (e.g. deserialised estimates) are
            // kept bit-exact so that save/load round-trips.
            let od = if (length - 1.0).abs() <= 1e-6 {
                clamped
            } else {
                linalg::normalize(clamped).ok_or(StainBasisError::Zero(index))?
            };
            normalized.push(Stain {
                name: stain.name,
                od,
            });
        }
        let vectors: Vec<Vec3> = normalized.iter().map(|s| s.od).collect();
        let min_angle = linalg::min_pairwise_angle_deg(&vectors);
        if vectors.len() > 1
            && (min_angle < MIN_BASIS_ANGLE_DEG || linalg::pseudo_inverse(&vectors).is_none())
        {
            return Err(StainBasisError::Degenerate(min_angle));
        }
        Ok(Self { stains: normalized })
    }

    /// The vectors of a built-in preset.
    pub fn preset(preset: StainPreset) -> Self {
        let stains = preset
            .definition()
            .iter()
            .map(|(name, od)| Stain::new(*name, *od))
            .collect();
        Self::new(stains).expect("built-in stain presets are valid")
    }

    /// Number of stains (1–3).
    pub fn len(&self) -> usize {
        self.stains.len()
    }

    /// Always `false`; a basis has at least one stain.
    pub fn is_empty(&self) -> bool {
        self.stains.is_empty()
    }

    pub fn stains(&self) -> &[Stain] {
        &self.stains
    }

    /// The unit OD vectors, in stain order.
    pub fn vectors(&self) -> Vec<[f32; 3]> {
        self.stains.iter().map(|s| s.od).collect()
    }

    /// Stain names, in stain order.
    pub fn names(&self) -> Vec<&str> {
        self.stains.iter().map(|s| s.name.as_str()).collect()
    }

    /// Smallest angle between any two stain vectors (90° for one stain).
    pub fn min_angle_deg(&self) -> f32 {
        linalg::min_pairwise_angle_deg(&self.vectors())
    }

    /// 2-norm condition number of the stain matrix.
    pub fn condition_number(&self) -> f32 {
        linalg::condition_number(&self.vectors())
    }

    /// Rows of the pseudo-inverse (`n` meaningful rows, the rest zero).
    pub fn pseudo_inverse(&self) -> [[f32; 3]; 3] {
        linalg::pseudo_inverse(&self.vectors()).expect("validated basis is invertible")
    }

    /// Largest angle between corresponding stains of `self` and `other`
    /// (`None` if the bases have different sizes).
    pub fn max_deviation_deg(&self, other: &StainBasis) -> Option<f32> {
        if self.len() != other.len() {
            return None;
        }
        Some(
            self.stains
                .iter()
                .zip(&other.stains)
                .map(|(a, b)| linalg::angle_deg(a.od, b.od))
                .fold(0.0f32, f32::max),
        )
    }

    /// Returns the preset whose vectors and names this basis matches exactly
    /// (within rounding), if any.
    pub fn matching_preset(&self) -> Option<StainPreset> {
        StainPreset::ALL.iter().copied().find(|preset| {
            let other = StainBasis::preset(*preset);
            other.len() == self.len()
                && other
                    .stains
                    .iter()
                    .zip(&self.stains)
                    .all(|(a, b)| linalg::angle_deg(a.od, b.od) < 0.01)
        })
    }
}

impl TryFrom<Vec<Stain>> for StainBasis {
    type Error = StainBasisError;

    fn try_from(stains: Vec<Stain>) -> Result<Self, Self::Error> {
        Self::new(stains)
    }
}

impl From<StainBasis> for Vec<Stain> {
    fn from(basis: StainBasis) -> Self {
        basis.stains
    }
}

/// Built-in stain vector presets (CLR-8).
///
/// Values are the published colour-deconvolution vectors of Ruifrok &
/// Johnston (2001) as distributed with G. Landini's ImageJ "Colour
/// Deconvolution" plugin and scikit-image, plus Macenko et al. (2009)'s
/// reference H&E matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum StainPreset {
    /// Hematoxylin & eosin (Ruifrok & Johnston; ImageJ "H&E").
    HematoxylinEosin,
    /// Hematoxylin & eosin, Macenko et al. reference matrix (`HERef`). This
    /// is the default normalisation target.
    HematoxylinEosinMacenko,
    /// Hematoxylin & DAB, for brightfield IHC (ImageJ "H DAB").
    HematoxylinDab,
    /// Hematoxylin, eosin & DAB (ImageJ "H&E DAB").
    HematoxylinEosinDab,
    /// Masson's trichrome: methyl blue & ponceau-fuchsin (ImageJ "Masson
    /// Trichrome").
    MassonTrichrome,
    /// Hematoxylin & periodic acid–Schiff.
    HematoxylinPas,
    /// Hematoxylin & AEC (3-amino-9-ethylcarbazole) IHC chromogen.
    HematoxylinAec,
}

impl StainPreset {
    pub const ALL: [StainPreset; 7] = [
        StainPreset::HematoxylinEosin,
        StainPreset::HematoxylinEosinMacenko,
        StainPreset::HematoxylinDab,
        StainPreset::HematoxylinEosinDab,
        StainPreset::MassonTrichrome,
        StainPreset::HematoxylinPas,
        StainPreset::HematoxylinAec,
    ];

    fn definition(self) -> &'static [(&'static str, [f32; 3])] {
        match self {
            Self::HematoxylinEosin => &[
                ("Hematoxylin", [0.644211, 0.716556, 0.266844]),
                ("Eosin", [0.092789, 0.954111, 0.283111]),
            ],
            Self::HematoxylinEosinMacenko => &[
                ("Hematoxylin", [0.5626, 0.7201, 0.4062]),
                ("Eosin", [0.2159, 0.8012, 0.5581]),
            ],
            Self::HematoxylinDab => &[
                ("Hematoxylin", [0.650, 0.704, 0.286]),
                ("DAB", [0.268, 0.570, 0.776]),
            ],
            Self::HematoxylinEosinDab => &[
                ("Hematoxylin", [0.650, 0.704, 0.286]),
                ("Eosin", [0.072, 0.990, 0.105]),
                ("DAB", [0.268, 0.570, 0.776]),
            ],
            Self::MassonTrichrome => &[
                ("Methyl blue", [0.799_510_7, 0.591_352_1, 0.105_286_67]),
                ("Ponceau-fuchsin", [0.099_971_59, 0.737_386_05, 0.668_032_6]),
            ],
            Self::HematoxylinPas => &[
                ("Hematoxylin", [0.644211, 0.716556, 0.266844]),
                ("PAS", [0.175411, 0.972178, 0.154589]),
            ],
            Self::HematoxylinAec => &[
                ("Hematoxylin", [0.650, 0.704, 0.286]),
                ("AEC", [0.2743, 0.6796, 0.6803]),
            ],
        }
    }

    /// The preset's stain vectors.
    pub fn basis(self) -> StainBasis {
        StainBasis::preset(self)
    }

    /// Short display name, e.g. `"H&E"` or `"H-DAB"`.
    pub fn label(self) -> &'static str {
        match self {
            Self::HematoxylinEosin => "H&E",
            Self::HematoxylinEosinMacenko => "H&E (Macenko reference)",
            Self::HematoxylinDab => "H-DAB",
            Self::HematoxylinEosinDab => "H-E-DAB",
            Self::MassonTrichrome => "Masson trichrome",
            Self::HematoxylinPas => "H-PAS",
            Self::HematoxylinAec => "H-AEC",
        }
    }

    /// Stable identifier used in files and on the command line.
    pub fn id(self) -> &'static str {
        match self {
            Self::HematoxylinEosin => "he",
            Self::HematoxylinEosinMacenko => "he-macenko",
            Self::HematoxylinDab => "h-dab",
            Self::HematoxylinEosinDab => "h-e-dab",
            Self::MassonTrichrome => "masson-trichrome",
            Self::HematoxylinPas => "h-pas",
            Self::HematoxylinAec => "h-aec",
        }
    }

    /// Parses [`StainPreset::id`] (case-insensitive).
    pub fn from_id(id: &str) -> Option<Self> {
        let id = id.trim().to_ascii_lowercase();
        Self::ALL.iter().copied().find(|p| p.id() == id)
    }

    /// `true` for the two-stain hematoxylin & eosin presets, the only stain
    /// model that Macenko/Vahadane normalisation applies to.
    pub fn is_hematoxylin_eosin(self) -> bool {
        matches!(self, Self::HematoxylinEosin | Self::HematoxylinEosinMacenko)
    }
}

/// What a slide's stain label says about how it was stained.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StainLabelKind {
    /// Hematoxylin & eosin.
    HematoxylinEosin,
    /// Brightfield immunohistochemistry (DAB or another chromogen).
    Immunohistochemistry,
    /// A named special stain (trichrome, PAS, Giemsa, ...).
    Special(String),
    /// No label, or a label that names no recognised stain.
    Unknown,
}

const IHC_TOKENS: &[&str] = &[
    "ihc",
    "dab",
    "immunohistochemistry",
    "immunostain",
    "immuno",
    "ki67",
    "ki-67",
    "mib1",
    "mib-1",
    "her2",
    "her-2",
    "pdl1",
    "pd-l1",
    "p53",
    "p16",
    "cd3",
    "cd4",
    "cd8",
    "cd20",
    "cd34",
    "cd45",
    "cd68",
    "ck7",
    "ck20",
    "pan-ck",
    "panck",
    "ae1/ae3",
    "sox10",
    "ttf1",
    "ttf-1",
    "gata3",
    "erg",
    "synaptophysin",
    "chromogranin",
];

fn tokens(label: &str) -> Vec<String> {
    label
        .to_ascii_lowercase()
        .split(|c: char| c.is_whitespace() || matches!(c, ',' | ';' | '|' | '(' | ')' | '_'))
        .filter(|t| !t.is_empty())
        .map(str::to_string)
        .collect()
}

/// Classifies a slide's stain label (e.g. `WsiProperties::stain`).
pub fn classify_stain_label(label: Option<&str>) -> StainLabelKind {
    let Some(label) = label.map(str::trim).filter(|l| !l.is_empty()) else {
        return StainLabelKind::Unknown;
    };
    let lower = label.to_ascii_lowercase();
    let toks = tokens(label);
    let has = |t: &str| toks.iter().any(|tok| tok == t);
    let has_sequence = |seq: &[&str]| {
        toks.windows(seq.len())
            .any(|w| w.iter().zip(seq).all(|(a, b)| a == b))
    };

    if lower.contains("hematoxylin and eosin")
        || lower.contains("haematoxylin and eosin")
        || lower.contains("hematoxylin & eosin")
        || lower.contains("haematoxylin & eosin")
        || has("h&e")
        || has("he")
        || has_sequence(&["h", "&", "e"])
        || has_sequence(&["h", "and", "e"])
    {
        return StainLabelKind::HematoxylinEosin;
    }
    if IHC_TOKENS.iter().any(|t| has(t)) || lower.contains("immunohisto") {
        return StainLabelKind::Immunohistochemistry;
    }
    for (needle, name) in [
        ("trichrome", "Masson trichrome"),
        ("masson", "Masson trichrome"),
        ("giemsa", "Giemsa"),
        ("reticulin", "Reticulin"),
        ("congo", "Congo red"),
        ("gieson", "Van Gieson"),
        ("elastic", "Elastic"),
        ("alcian", "Alcian blue"),
        ("perls", "Perls' Prussian blue"),
        ("gram", "Gram"),
        ("grocott", "Grocott methenamine silver"),
        ("gms", "Grocott methenamine silver"),
        ("mucicarmine", "Mucicarmine"),
        ("ziehl", "Ziehl–Neelsen"),
    ] {
        if lower.contains(needle) {
            return StainLabelKind::Special(name.to_string());
        }
    }
    if has("pas") || lower.contains("periodic acid") {
        return StainLabelKind::Special("PAS".to_string());
    }
    StainLabelKind::Unknown
}

/// The preset that best matches a slide's stain label, if any.
pub fn suggested_preset(label: Option<&str>) -> Option<StainPreset> {
    match classify_stain_label(label) {
        StainLabelKind::HematoxylinEosin => Some(StainPreset::HematoxylinEosin),
        StainLabelKind::Immunohistochemistry => Some(StainPreset::HematoxylinDab),
        StainLabelKind::Special(name) if name == "Masson trichrome" => {
            Some(StainPreset::MassonTrichrome)
        }
        StainLabelKind::Special(name) if name == "PAS" => Some(StainPreset::HematoxylinPas),
        _ => None,
    }
}

/// A user-facing warning when the selected stain model does not fit the
/// slide's stain label (CLR-8), e.g. H&E normalisation on an IHC slide,
/// which would turn DAB brown into eosin pink.
///
/// `uses_he_model` is `true` when H&E normalisation or the H&E deconvolution
/// preset is active. Returns `None` when there is nothing to warn about
/// (including unlabelled slides).
pub fn stain_model_warning(label: Option<&str>, uses_he_model: bool) -> Option<String> {
    if !uses_he_model {
        return None;
    }
    match classify_stain_label(label) {
        StainLabelKind::Immunohistochemistry => Some(format!(
            "This slide is labelled \"{}\" (immunohistochemistry), but the H&E stain model \
             is selected; DAB and other chromogens will be shown in the wrong colours. \
             Use the H-DAB stain preset instead.",
            label.unwrap_or_default().trim()
        )),
        StainLabelKind::Special(name) => Some(format!(
            "This slide is labelled as a {name} stain, but the H&E stain model is selected; \
             its colours cannot be represented by hematoxylin and eosin."
        )),
        StainLabelKind::HematoxylinEosin | StainLabelKind::Unknown => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_are_unit_length_and_well_conditioned() {
        for preset in StainPreset::ALL {
            let basis = preset.basis();
            for stain in basis.stains() {
                let n = linalg::norm(stain.od);
                assert!(
                    (n - 1.0).abs() < 1e-5,
                    "{preset:?} {}: |v| = {n}",
                    stain.name
                );
                assert!(stain.od.iter().all(|&v| v >= 0.0));
            }
            assert!(
                basis.min_angle_deg() > 15.0,
                "{preset:?}: {}",
                basis.min_angle_deg()
            );
            assert_eq!(StainPreset::from_id(preset.id()), Some(preset));
            assert_eq!(basis.matching_preset(), Some(preset));
        }
    }

    #[test]
    fn macenko_reference_vectors_are_22_degrees_apart() {
        // Documented so the default fit checks stay compatible with the
        // reference: Macenko's own H and E vectors are only ~22° apart.
        let angle = StainPreset::HematoxylinEosinMacenko.basis().min_angle_deg();
        assert!((angle - 22.3).abs() < 0.3, "{angle}");
    }

    #[test]
    fn custom_basis_validation() {
        assert_eq!(StainBasis::new(vec![]), Err(StainBasisError::Empty));
        let v = Stain::new("a", [0.6, 0.7, 0.3]);
        assert!(matches!(
            StainBasis::new(vec![v.clone(), v.clone()]),
            Err(StainBasisError::Degenerate(_))
        ));
        assert_eq!(
            StainBasis::new(vec![Stain::new("neg", [-0.5, 0.7, 0.3])]),
            Err(StainBasisError::Negative(0))
        );
        assert_eq!(
            StainBasis::new(vec![Stain::new("zero", [0.0; 3])]),
            Err(StainBasisError::Zero(0))
        );
        let basis = StainBasis::new(vec![Stain::new("x", [2.0, 0.0, 0.0])]).unwrap();
        assert_eq!(basis.stains()[0].od, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn basis_serde_round_trip_validates() {
        let basis = StainPreset::HematoxylinDab.basis();
        let json = serde_json::to_string(&basis).unwrap();
        let back: StainBasis = serde_json::from_str(&json).unwrap();
        assert_eq!(back, basis);
        let bad = r#"[{"name":"a","od":[0.6,0.7,0.3]},{"name":"b","od":[0.6,0.7,0.3]}]"#;
        assert!(serde_json::from_str::<StainBasis>(bad).is_err());
    }

    #[test]
    fn stain_labels_are_classified() {
        assert_eq!(
            classify_stain_label(Some("H&E")),
            StainLabelKind::HematoxylinEosin
        );
        assert_eq!(
            classify_stain_label(Some("Hematoxylin and Eosin")),
            StainLabelKind::HematoxylinEosin
        );
        assert_eq!(
            classify_stain_label(Some("IHC")),
            StainLabelKind::Immunohistochemistry
        );
        assert_eq!(
            classify_stain_label(Some("Ki-67 (MIB-1)")),
            StainLabelKind::Immunohistochemistry
        );
        assert_eq!(
            classify_stain_label(Some("Masson trichrome")),
            StainLabelKind::Special("Masson trichrome".into())
        );
        assert_eq!(
            classify_stain_label(Some("PAS")),
            StainLabelKind::Special("PAS".into())
        );
        assert_eq!(classify_stain_label(None), StainLabelKind::Unknown);
        // Tokens must match whole words.
        assert_eq!(
            classify_stain_label(Some("Herbert")),
            StainLabelKind::Unknown
        );
        assert_eq!(classify_stain_label(Some("theme")), StainLabelKind::Unknown);
    }

    #[test]
    fn ihc_slides_get_a_warning_for_the_he_model() {
        let warning = stain_model_warning(Some("IHC"), true).unwrap();
        assert!(warning.contains("H-DAB"));
        assert!(stain_model_warning(Some("IHC"), false).is_none());
        assert!(stain_model_warning(Some("H&E"), true).is_none());
        assert!(stain_model_warning(None, true).is_none());
        assert_eq!(
            suggested_preset(Some("CD20")),
            Some(StainPreset::HematoxylinDab)
        );
    }
}
