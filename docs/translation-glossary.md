# Translation glossary

The terms eov and its plugins use, what they mean, and how to treat them. It
keeps translations consistent across the application and the plugins, and
across languages. Each language's chosen translations are listed in
[`glossary/<language>.tsv`](glossary/) (term, translation, note); add to them
when you settle a term, and follow them when you translate.

See [translating.md](translating.md) for how catalogs are edited and
submitted.

## General rules

- **Audience.** Pathologists, biomedical researchers, educators and students
  using desktop scientific software. Use the terminology of histopathology,
  microscopy and digital pathology as it is used in your language's
  textbooks, journals and laboratory software — not literal renderings of
  the English.
- **Register.** Write as established professional desktop software in your
  language does (operating systems, scientific and medical software). In
  languages with a polite and a familiar "you", use the polite form unless
  the platforms' own style guides in your language prefer otherwise; prefer
  impersonal or infinitive constructions for buttons and menu items where
  that is the convention.
- **Buttons, menu items, titles** follow your language's UI conventions
  (imperative or infinitive, capitalisation). English uses Title Case for
  menu items and buttons; most languages do not.
- **Placeholders** (`{}`, `{0}`, `{name}`, `{n}`) are replaced by values. Keep
  every one of them, add none, and move them where your grammar needs them.
  To put two `{}` in another order, number them: the first `{}` of the
  English is `{0}`, the second `{1}` (`{} of {}` → `{1} 中的 {0}`).
  `{n}` is the number of a text with plural forms. `{{` and `}}` are literal
  braces.
- **Plural forms.** A text with a number has one form per plural category of
  your language (see the catalog's `Plural-Forms` header). Every form that a
  number other than one can take must show `{n}`.
- **Ellipsis.** "…" at the end of a menu item means it opens a dialog; keep it
  (use your language's form of the ellipsis if it differs).
- **Units and numbers** stay as written: µm, mm, px, MB, GiB, %, °, ×. Use
  your language's spacing conventions (French puts a space before `%`), but
  never change the unit. Decimal separators are handled by eov.
- **Keyboard shortcuts** (Ctrl+B, Shift, Alt, Esc, F1, Enter, Tab, arrows)
  stay as written; they match the keys.
- **Do not translate** (see below) names of formats, products, methods and
  identifiers, and anything that comes from a slide or a file.

## Safety and intended use

These texts have legal and clinical weight. Translate them completely and
literally enough that nothing is weakened, softened or omitted; when in
doubt, prefer the more explicit wording. Never shorten them to fit.

| English | Meaning |
|---|---|
| For research and education. Not a medical device; not for primary diagnosis. | eov is not certified medical software and must not be the basis of a diagnosis. |
| research use only (RUO) | Results must not be used for clinical decisions. Use the regulatory phrase of your language/region if one exists (e.g. the term printed on RUO reagents). |
| synthetic, predicted, generated (image, channel, stain) | The pixels were computed by an algorithm (normalisation, deconvolution, a neural network), not captured by the scanner. Never translate as "artificial" in a sense that could read as "fake" in a pejorative way, and never as "simulated" if your language implies a game; "computed/predicted/synthesised" are right. |
| primary diagnosis | The first, final diagnosis of a case (regulatory term). |
| readout, region readout (Cellaris) | Statistics (mean, spread, density r²) of a model's *predicted* values over a region. It is computed from predictions and is not a measurement: do not use words for measuring, an assay readout or measured signal (German not "Messung"); a word for a summary or evaluation of predicted values is right. |

## Terms

| English | Meaning and guidance |
|---|---|
| slide | A **glass microscope slide** and its scan. Never a presentation slide or a sliding part. Use the histopathology term for the specimen slide (de: Objektträger/Präparat, fr: lame, es: portaobjetos/preparación, it: vetrino, ja: スライド/標本, ru: препарат/стекло…). |
| whole-slide image (WSI) | A scan of a complete slide at microscopic resolution. Keep the abbreviation WSI. |
| slide viewer, viewer | The application that shows slides. |
| tab | One open slide in a pane (like a browser tab). |
| pane | One of several side-by-side viewing areas of the window. |
| view | What a pane shows: position, magnification, rotation. |
| viewport | The area of a pane where the slide is drawn. |
| minimap | The small overview image of the whole slide with the visible area marked. |
| scale bar | The bar showing a physical length (µm, mm) on the image. |
| HUD | The small panels that float over the slide (heads-up display). Translate the idea ("overlay panel") or keep HUD if your language's software does. |
| toolbar, sidebar, status bar | Standard UI parts; use your platform's terms. |
| zoom, magnification (20×) | "Magnification" is the objective-equivalent optical magnification; "zoom" is the screen scale. 20× stays "20×". |
| objective | The microscope objective (e.g. 40×). |
| pixel size, MPP (µm per pixel), calibration | The physical size of one image pixel; MPP stays MPP. |
| level, pyramid level | One resolution of the image pyramid stored in the slide file. |
| tile | A square piece of the image loaded at a time; also a dataset patch. |
| cache | Memory or disk space holding loaded tiles. |
| region of interest (ROI) | A rectangle the user draws to export, measure or analyse. Keep ROI as the abbreviation. |
| measurement, measure (tool) | Distances and areas measured on the slide. Only for real measurements (eov's measure tool, measured IF); Cellaris's region readout of predicted values is not a measurement. |
| compute (predictions, a readout) | Calculating values with a model ("Computing: 3 of 10 cells", "Compute a readout of a large region?"); never "measure". |
| cell (Cellaris) | One square of the grid a slide is predicted in (41.6 µm with the shipped model), not a biological cell. |
| tissue block (Cellaris) | A 5 µm square of tissue within a Cellaris cell, not a paraffin block. |
| annotation | A drawn mark with a meaning (point, region, line, text) — not a free-text comment. |
| class (annotation class) | A category such as "Tumour" or "Stroma" given to annotations. |
| cell count, counting | Counting cells by placing points. |
| high-power field (HPF) | The area seen with a 40× objective, a standard counting area in pathology. Use the established term (de: Hauptgesichtsfeld/HPF, fr: champ à fort grossissement…). |
| H&E, hematoxylin, eosin | The standard stain and its two dyes. Use the standard spelling of your language (haematoxylin/hematoxylin). |
| IHC, DAB | Immunohistochemistry, and its brown chromogen DAB. Keep the abbreviations. |
| stain normalization (Macenko, Vahadane) | Adjusting a slide's colours to a reference; Macenko and Vahadane are method names (do not translate). |
| color deconvolution | Separating the dyes of a stain into channels (Ruifrok–Johnston). |
| channel | A fluorescence or computed image channel. |
| multiplex immunofluorescence (mIF) | Several fluorescent markers on one slide. |
| marker | A protein stained or predicted in a channel (CD3, Ki-67, PanCK…); marker names are never translated. |
| focal plane (Z), time point (T) | Z-stack plane and time series index. |
| heat map | A coloured overlay of values over the slide. |
| overlay, layer | Graphics drawn over the slide; a layer groups them. |
| opacity | How opaque an overlay is (0–100 %). |
| tissue QC, pen marks, folds, blur | Quality control of the scanned tissue and its common artefacts. |
| tissue detection, tissue mask | Finding where tissue (not glass) is. |
| dataset, patch extraction, stride | Cutting a slide into small images for machine learning; the stride is the step between patches. |
| export, save a copy, convert | Writing images or slides to files. |
| de-identify | Removing data that could identify a patient (label image, metadata). Use the term of your language's data-protection practice. |
| label image, macro image, thumbnail | The photo of the slide's paper label, of the whole glass slide, and the small preview. |
| bookmark | A saved view of a slide. |
| session | The slides and views that were open, restored at the next start. |
| plugin, package (.eop) | An extension of eov and the file it is installed from. |
| model, model store | A machine-learning model file, and eov's storage for them. |
| inference | Running a model on images. |
| render, renderer, backend (CPU, GPU) | Drawing the image; CPU and GPU stay as they are. |
| filtering (bilinear, trilinear, Lanczos) | Texture interpolation methods; Lanczos is a name. |
| sharpness, gamma, brightness, contrast | Image adjustments. |
| theme (dark, light, high contrast) | The window's colour scheme. |
| Home screen | The start page shown when no slide is open. |
| Check for Updates… | Asks before going online; never imply automatic background updates. |

## Never translate

eov, OpenSlide, Slint, DICOM, DICOMweb, TIFF, BigTIFF, OME-TIFF, OME-Zarr,
SVS, NDPI, VMS, VMU, MRXS (MIRAX), SCN, BIF, CZI, QPTIFF, iSyntax, VSI, PNG,
JPEG, GeoJSON, CSV, JSON, TOML, XML, HDF5, Zarr, ONNX, ONNX Runtime, CUDA,
Vulkan, Metal, wgpu, GPU, CPU, ROI (as an abbreviation), WSI, MPP, HPF (as an
abbreviation), H&E, IHC, DAB, RGB, RGBA, ICC, sRGB, QuPath, ASAP, Aperio,
ImageScope, Hamamatsu, NDPA, Leica, Philips, Zeiss, Ventana, CLAM,
TIAToolbox, slideflow, HistoVAE, CAMELYON17, Cellaris, Macenko, Vahadane,
Ruifrok, Lanczos, GitHub, Flatpak, AppImage, Homebrew, Windows, macOS,
Linux, file extensions (`.svs`, `.eop`), paths, URLs, command-line options
(`--language`), configuration keys (`config.toml`, `language = "de"`),
marker and channel names, and every value that comes from a slide, a file or
the user.
