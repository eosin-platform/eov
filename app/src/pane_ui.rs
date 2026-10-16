use crate::gpu::GpuRenderer;
use crate::render_pool::CachedCpuFrame;
use crate::state::PaneId;
use crate::{PaneViewData, SeriesItemData, TabData};
use slint::{Image, VecModel};
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Default)]
pub(crate) struct PaneRenderCacheEntry {
    pub(crate) content: Option<Image>,
    pub(crate) minimap_thumbnail: Option<Image>,
    /// How `minimap_thumbnail` is turned: the pane view's nearest quarter
    /// turn.
    pub(crate) minimap_orientation: eov_common::viewport::Orientation,
    /// The colour profile `minimap_thumbnail` is converted from, if any.
    pub(crate) minimap_color: Option<u64>,
    pub(crate) cpu_frame: Option<CachedCpuFrame>,
    /// Reused for the frames shown while the view moves (CPU backend).
    pub(crate) preview_buffer: Vec<u8>,
}

impl Clone for PaneRenderCacheEntry {
    fn clone(&self) -> Self {
        Self {
            content: self.content.clone(),
            minimap_thumbnail: self.minimap_thumbnail.clone(),
            minimap_orientation: self.minimap_orientation,
            minimap_color: self.minimap_color,
            cpu_frame: None,
            preview_buffer: Vec::new(),
        }
    }
}

#[derive(Clone)]
pub(crate) struct PaneUiModels {
    pub(crate) tabs: Rc<VecModel<TabData>>,
}

impl Default for PaneUiModels {
    fn default() -> Self {
        Self {
            tabs: Rc::new(VecModel::default()),
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
    /// The stable pane id of each entry of the per-pane caches, in order.
    static PANE_CACHE_UIDS: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
    /// New pane → the pane whose picture it shows until it first renders.
    static PANE_SEEDS: RefCell<Vec<(u32, u32)>> = const { RefCell::new(Vec::new()) };
}

/// Re-orders the per-pane caches to follow the panes' stable ids (`uids`,
/// in pane order): a pane keeps its picture, minimap and tab models when
/// panes are inserted, moved or removed around it, removed panes free
/// theirs, and new panes start from their seed (see
/// [`insert_pane_ui_state`]) or empty (UI-2, UI-16).
pub(crate) fn sync_pane_caches(
    uids: &[u32],
    cache: &mut Vec<PaneRenderCacheEntry>,
    models: &mut Vec<PaneUiModels>,
) {
    let known = PANE_CACHE_UIDS.with(|known| known.borrow().clone());
    let seeds = PANE_SEEDS.with(|seeds| std::mem::take(&mut *seeds.borrow_mut()));
    if known == uids && cache.len() == uids.len() && models.len() == uids.len() {
        return;
    }
    // Entries beyond what was recorded belong to the panes at their
    // position (caches grow with the panes).
    let uid_at = |index: usize| known.get(index).or_else(|| uids.get(index)).copied();

    let mut old_entries: std::collections::HashMap<u32, PaneRenderCacheEntry> =
        std::collections::HashMap::new();
    for (index, entry) in cache.drain(..).enumerate() {
        match uid_at(index) {
            Some(uid) if !old_entries.contains_key(&uid) => {
                old_entries.insert(uid, entry);
            }
            _ => recycle(entry),
        }
    }
    let mut seeded: std::collections::HashMap<u32, PaneRenderCacheEntry> = seeds
        .iter()
        .filter_map(|(new, source)| Some((*new, old_entries.get(source)?.clone())))
        .collect();
    *cache = uids
        .iter()
        .map(|uid| {
            old_entries
                .remove(uid)
                .or_else(|| seeded.remove(uid))
                .unwrap_or_default()
        })
        .collect();
    for (_, entry) in old_entries {
        recycle(entry);
    }

    let mut old_models: std::collections::HashMap<u32, PaneUiModels> =
        std::collections::HashMap::new();
    for (index, model) in models.drain(..).enumerate() {
        if let Some(uid) = uid_at(index) {
            old_models.entry(uid).or_insert(model);
        }
    }
    *models = uids
        .iter()
        .map(|uid| old_models.remove(uid).unwrap_or_default())
        .collect();

    PANE_CACHE_UIDS.with(|known| *known.borrow_mut() = uids.to_vec());
}

fn recycle(mut entry: PaneRenderCacheEntry) {
    if let Some(frame) = entry.cpu_frame.take()
        && let Some(pool) = crate::render_pool::global()
    {
        pool.recycle_buffer(frame.pixels);
        pool.recycle_buffer(frame.source);
    }
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
}

pub(crate) fn set_cached_pane_cpu_result(pane: PaneId, image: Image, frame: CachedCpuFrame) {
    with_pane_render_cache(pane.0 + 1, |cache| {
        if let Some(old_frame) = cache[pane.0].cpu_frame.take()
            && let Some(pool) = crate::render_pool::global()
        {
            pool.recycle_buffer(old_frame.pixels);
            pool.recycle_buffer(old_frame.source);
        }
        cache[pane.0].content = Some(image);
        cache[pane.0].cpu_frame = Some(frame);
    });
}

pub(crate) fn set_cached_pane_minimap(
    pane: PaneId,
    image: Option<Image>,
    orientation: eov_common::viewport::Orientation,
    color: Option<u64>,
) {
    with_pane_render_cache(pane.0 + 1, |cache| {
        cache[pane.0].minimap_thumbnail = image;
        cache[pane.0].minimap_orientation = orientation.quarter_part();
        cache[pane.0].minimap_color = color;
    });
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

/// Lets the pane `new_uid` show `source_uid`'s picture until it renders
/// (a split shows the same view at first). Applied by the next
/// [`sync_pane_caches`].
pub(crate) fn insert_pane_ui_state(new_uid: Option<u32>, source_uid: Option<u32>) {
    if let (Some(new_uid), Some(source_uid)) = (new_uid, source_uid) {
        PANE_SEEDS.with(|seeds| seeds.borrow_mut().push((new_uid, source_uid)));
    }
}

pub(crate) fn clear_cached_pane(pane: PaneId) {
    with_pane_render_cache(pane.0 + 1, |cache| {
        if let Some(old_frame) = cache[pane.0].cpu_frame.take()
            && let Some(pool) = crate::render_pool::global()
        {
            pool.recycle_buffer(old_frame.pixels);
            pool.recycle_buffer(old_frame.source);
        }
        cache[pane.0] = PaneRenderCacheEntry::default();
    });
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
