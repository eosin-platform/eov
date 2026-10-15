//! Sizing the tile cache for the display (WSI-4).
//!
//! Unless `--cache-size`/`--max-tiles` fix them, the tile cache limits
//! follow the window: two worst-case frames per visible pane
//! ([`CacheBudget::for_display`]). OpenSlide's own decode cache is sized
//! from the tile budget.

use eov_common::wsi::{
    openslide_cache_capacity, recommended_openslide_cache_bytes, set_openslide_cache_capacity,
};
use eov_common::{CacheBudget, TileCache};
use std::cell::{Cell, RefCell};

/// Window size assumed before the window is laid out.
const INITIAL_SIZE: (u32, u32) = (1920, 1080);

struct Sizing {
    fixed_bytes: Option<usize>,
    fixed_tiles: Option<usize>,
    /// Window width, height and visible panes last sized for.
    last: Cell<Option<(u32, u32, u32)>>,
}

impl Sizing {
    fn budget(&self, width: u32, height: u32, panes: u32) -> CacheBudget {
        let display = CacheBudget::for_display(width, height, panes);
        CacheBudget {
            max_bytes: self.fixed_bytes.unwrap_or(display.max_bytes),
            max_tiles: self.fixed_tiles.unwrap_or(display.max_tiles),
        }
    }
}

thread_local! {
    static SIZING: RefCell<Option<Sizing>> = const { RefCell::new(None) };
}

/// The initial budget; call once on the UI thread before creating the
/// cache. `None` limits follow the display.
pub(crate) fn init(fixed_bytes: Option<usize>, fixed_tiles: Option<usize>) -> CacheBudget {
    let sizing = Sizing {
        fixed_bytes,
        fixed_tiles,
        last: Cell::new(None),
    };
    let budget = sizing.budget(INITIAL_SIZE.0, INITIAL_SIZE.1, 1);
    apply_openslide_capacity(budget.max_bytes);
    SIZING.with(|slot| *slot.borrow_mut() = Some(sizing));
    budget
}

/// Re-sizes the cache for a window of `width` x `height` physical pixels
/// showing `panes` panes, when any of them changed. Cheap to call every
/// frame.
pub(crate) fn update(cache: &TileCache, width: u32, height: u32, panes: u32) {
    if width == 0 || height == 0 {
        return;
    }
    SIZING.with(|slot| {
        let slot = slot.borrow();
        let Some(sizing) = slot.as_ref() else {
            return;
        };
        let key = (width, height, panes.max(1));
        if sizing.last.get() == Some(key) {
            return;
        }
        sizing.last.set(Some(key));
        if sizing.fixed_bytes.is_some() && sizing.fixed_tiles.is_some() {
            return;
        }
        let budget = sizing.budget(key.0, key.1, key.2);
        cache.set_limits(budget.max_tiles, budget.max_bytes);
        apply_openslide_capacity(budget.max_bytes);
    });
}

/// Sizes OpenSlide's decode cache from the tile budget, ignoring small
/// changes (replacing the cache drops what it holds).
fn apply_openslide_capacity(tile_cache_bytes: usize) {
    let wanted = recommended_openslide_cache_bytes(tile_cache_bytes);
    let current = openslide_cache_capacity();
    if current != 0 && wanted.abs_diff(current) < current / 2 {
        return;
    }
    if let Err(err) = set_openslide_cache_capacity(wanted) {
        tracing::warn!("Could not size the OpenSlide cache: {err}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_limits_win_over_the_display() {
        let sizing = Sizing {
            fixed_bytes: Some(64 << 20),
            fixed_tiles: None,
            last: Cell::new(None),
        };
        let budget = sizing.budget(3840, 2160, 2);
        assert_eq!(budget.max_bytes, 64 << 20);
        assert_eq!(
            budget.max_tiles,
            CacheBudget::for_display(3840, 2160, 2).max_tiles
        );
    }

    #[test]
    fn display_budget_grows_with_panes() {
        let sizing = Sizing {
            fixed_bytes: None,
            fixed_tiles: None,
            last: Cell::new(None),
        };
        assert!(sizing.budget(1920, 1080, 2).max_bytes >= sizing.budget(1920, 1080, 1).max_bytes);
    }
}
