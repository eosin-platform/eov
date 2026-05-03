use crate::gpu::GpuRenderer;
use crate::render_pool::CachedCpuFrame;
use crate::state::PaneId;
use crate::{MeasurementLine, PaneViewData, SeriesItemData, TabData};
use slint::{Image, VecModel};
use std::cell::RefCell;
use std::rc::Rc;
use tracing::debug;

#[derive(Default)]
pub(crate) struct PaneRenderCacheEntry {
    pub(crate) content: Option<Image>,
    pub(crate) minimap_thumbnail: Option<Image>,
    pub(crate) cpu_frame: Option<CachedCpuFrame>,
    pub(crate) preview_buffer: Vec<u8>,
    pub(crate) preview_width: u32,
    pub(crate) preview_height: u32,
}

impl Clone for PaneRenderCacheEntry {
    fn clone(&self) -> Self {
        Self {
            content: self.content.clone(),
            minimap_thumbnail: self.minimap_thumbnail.clone(),
            cpu_frame: None,
            preview_buffer: Vec::new(),
            preview_width: 0,
            preview_height: 0,
        }
    }
}

#[derive(Clone)]
pub(crate) struct PaneUiModels {
    pub(crate) tabs: Rc<VecModel<TabData>>,
    pub(crate) measurements: Rc<VecModel<MeasurementLine>>,
}

impl Default for PaneUiModels {
    fn default() -> Self {
        Self {
            tabs: Rc::new(VecModel::default()),
            measurements: Rc::new(VecModel::default()),
        }
    }
}

thread_local! {
    static GPU_RENDERER_HANDLE: RefCell<Option<Rc<RefCell<GpuRenderer>>>> = const { RefCell::new(None) };
    static PANE_RENDER_CACHE: RefCell<Vec<PaneRenderCacheEntry>> = const { RefCell::new(Vec::new()) };
    static PANE_VIEW_MODEL: RefCell<Rc<VecModel<PaneViewData>>> = RefCell::new(Rc::new(VecModel::default()));
    static PANE_UI_MODELS: RefCell<Vec<PaneUiModels>> = const { RefCell::new(Vec::new()) };
    static SERIES_ITEMS_MODEL: RefCell<Rc<VecModel<SeriesItemData>>> = RefCell::new(Rc::new(VecModel::default()));
    static SERIES_ITEMS_REVISION: RefCell<u64> = const { RefCell::new(0) };
}

pub(crate) fn pane_from_index(index: i32) -> PaneId {
    PaneId(index.max(0) as usize)
}

pub(crate) fn series_items_model() -> Rc<VecModel<SeriesItemData>> {
    SERIES_ITEMS_MODEL.with(|model| model.borrow().clone())
}

pub(crate) fn series_items_revision() -> u64 {
    SERIES_ITEMS_REVISION.with(|revision| *revision.borrow())
}

pub(crate) fn set_series_items_revision(revision: u64) {
    SERIES_ITEMS_REVISION.with(|stored| {
        *stored.borrow_mut() = revision;
    });
}

pub(crate) fn with_pane_render_cache<T>(
    pane_count: usize,
    f: impl FnOnce(&mut Vec<PaneRenderCacheEntry>) -> T,
) -> T {
    PANE_RENDER_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if cache.len() < pane_count {
            cache.resize_with(pane_count, PaneRenderCacheEntry::default);
        }
        f(&mut cache)
    })
}

pub(crate) fn set_cached_pane_content(pane: PaneId, image: Image) {
    with_pane_render_cache(pane.0 + 1, |cache| {
        cache[pane.0].content = Some(image);
    });

    if pane.0 == 1 {
        debug!(pane = pane.0, "set pane cached content");
    }
}

pub(crate) fn set_cached_pane_cpu_result(pane: PaneId, image: Image, frame: CachedCpuFrame) {
    with_pane_render_cache(pane.0 + 1, |cache| {
        if let Some(old_frame) = cache[pane.0].cpu_frame.take()
            && let Some(pool) = crate::render_pool::global()
        {
            pool.recycle_buffer(old_frame.pixels);
        }
        cache[pane.0].content = Some(image);
        cache[pane.0].cpu_frame = Some(frame);
        cache[pane.0].preview_buffer.clear();
        cache[pane.0].preview_width = 0;
        cache[pane.0].preview_height = 0;
    });
}

pub(crate) fn set_cached_pane_minimap(pane: PaneId, image: Option<Image>) {
    let has_minimap = image.is_some();
    with_pane_render_cache(pane.0 + 1, |cache| {
        cache[pane.0].minimap_thumbnail = image;
    });

    if pane.0 == 1 {
        debug!(pane = pane.0, has_minimap, "set pane cached minimap");
    }
}

pub(crate) fn with_pane_view_model<T>(f: impl FnOnce(&Rc<VecModel<PaneViewData>>) -> T) -> T {
    PANE_VIEW_MODEL.with(|model| {
        let model = model.borrow();
        f(&model)
    })
}

pub(crate) fn with_pane_ui_models<T>(
    pane_count: usize,
    f: impl FnOnce(&mut Vec<PaneUiModels>) -> T,
) -> T {
    PANE_UI_MODELS.with(|models| {
        let mut models = models.borrow_mut();
        if models.len() < pane_count {
            models.resize_with(pane_count, PaneUiModels::default);
        }
        f(&mut models)
    })
}

pub(crate) fn reset_pane_ui_state() {
    PANE_RENDER_CACHE.with(|cache| cache.borrow_mut().clear());
    PANE_UI_MODELS.with(|models| models.borrow_mut().clear());
    SERIES_ITEMS_REVISION.with(|revision| {
        *revision.borrow_mut() = 0;
    });
    PANE_VIEW_MODEL.with(|model| {
        let model = model.borrow_mut();
        model.clear();
    });
}

pub(crate) fn insert_pane_ui_state(new_pane: PaneId, source_pane: Option<PaneId>) {
    PANE_RENDER_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let insert_index = new_pane.0.min(cache.len());
        let entry = source_pane
            .and_then(|pane| cache.get(pane.0).cloned())
            .unwrap_or_default();
        cache.insert(insert_index, entry);
    });

    PANE_UI_MODELS.with(|models| {
        let mut models = models.borrow_mut();
        let insert_index = new_pane.0.min(models.len());
        models.insert(insert_index, PaneUiModels::default());
    });
}

pub(crate) fn clear_cached_pane(pane: PaneId) {
    with_pane_render_cache(pane.0 + 1, |cache| {
        if let Some(old_frame) = cache[pane.0].cpu_frame.take()
            && let Some(pool) = crate::render_pool::global()
        {
            pool.recycle_buffer(old_frame.pixels);
        }
        cache[pane.0] = PaneRenderCacheEntry::default();
    });

    if pane.0 == 1 {
        debug!(pane = pane.0, "cleared pane cache");
    }
}

pub(crate) fn set_gpu_renderer_handle(renderer: Rc<RefCell<GpuRenderer>>) {
    GPU_RENDERER_HANDLE.with(|handle| {
        *handle.borrow_mut() = Some(renderer);
    });
}

pub(crate) fn with_gpu_renderer<R>(f: impl FnOnce(&Rc<RefCell<GpuRenderer>>) -> R) -> Option<R> {
    GPU_RENDERER_HANDLE.with(|handle| {
        let handle = handle.borrow();
        handle.as_ref().map(f)
    })
}
