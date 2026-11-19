//! Similar photos and bursts: photos taken close together that look alike,
//! suggested as groups to choose from. Nothing here deletes, moves, or
//! stars a photo; the groups are only shown.
//!
//! How a photo looks is a 64-bit difference hash of its thumbnail: the
//! thumbnail in grey at 9 × 8, each bit saying whether a pixel is brighter
//! than the one to its right. Photos that look alike differ in few bits.

use image::{imageops, DynamicImage, RgbaImage};

/// A photo as the grouping sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shot {
    pub id: i64,
    /// When it was taken, in seconds; 0 when unknown.
    pub taken_at: i64,
    pub look: u64,
}

/// How far apart, in seconds, two shots of one group can be by default.
/// Five minutes, matching the library's "Within 5 minutes" grouping label.
pub const DEFAULT_WINDOW: i64 = 5 * 60;
/// How many of the 64 bits two shots of one group can differ in by default.
pub const DEFAULT_DISTANCE: u32 = 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StackMode {
    Time,
    LookAlike,
}

/// The difference hash of a thumbnail.
pub fn look(thumb: &RgbaImage) -> u64 {
    let grey = DynamicImage::ImageRgba8(thumb.clone()).into_luma8();
    let small = imageops::resize(&grey, 9, 8, imageops::FilterType::Triangle);
    let mut bits = 0u64;
    for y in 0..8 {
        for x in 0..8 {
            bits <<= 1;
            if small.get_pixel(x, y)[0] > small.get_pixel(x + 1, y)[0] {
                bits |= 1;
            }
        }
    }
    bits
}

/// How many bits two looks differ in, from 0 (alike) to 64.
pub fn distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Groups of two or more shots, in the order taken: each shot joins the
/// group of the one before it when it was taken within `window` seconds of
/// it and looks within `max_distance` of it. Shots with no time are left
/// out, as a time is what makes a burst.
pub fn group(shots: &[Shot], window: i64, max_distance: u32) -> Vec<Vec<i64>> {
    let mut timed: Vec<&Shot> = shots.iter().filter(|shot| shot.taken_at > 0).collect();
    timed.sort_by_key(|shot| (shot.taken_at, shot.id));
    let mut groups: Vec<Vec<i64>> = Vec::new();
    let mut current: Vec<i64> = Vec::new();
    let mut previous: Option<&Shot> = None;
    for shot in timed {
        let joins = previous.is_some_and(|before| shot.taken_at - before.taken_at <= window && distance(shot.look, before.look) <= max_distance);
        if !joins && current.len() > 1 {
            groups.push(std::mem::take(&mut current));
        }
        if !joins {
            current.clear();
        }
        current.push(shot.id);
        previous = Some(shot);
    }
    if current.len() > 1 {
        groups.push(current);
    }
    groups
}

pub fn group_by_time(shots: &[Shot], window: i64) -> Vec<Vec<i64>> {
    let mut timed: Vec<&Shot> = shots.iter().filter(|shot| shot.taken_at > 0).collect();
    timed.sort_by_key(|shot| (shot.taken_at, shot.id));
    let mut groups = Vec::new();
    let mut current = Vec::new();
    let mut previous: Option<&Shot> = None;
    for shot in timed {
        let separated = previous.is_some_and(|before| shot.taken_at - before.taken_at > window);
        if separated && current.len() > 1 {
            groups.push(std::mem::take(&mut current));
        }
        if separated {
            current.clear();
        }
        current.push(shot.id);
        previous = Some(shot);
    }
    if current.len() > 1 {
        groups.push(current);
    }
    groups
}

pub fn group_by_look(shots: &[Shot], max_distance: u32) -> Vec<Vec<i64>> {
    let mut ordered: Vec<&Shot> = shots.iter().collect();
    ordered.sort_by_key(|shot| (shot.taken_at == 0, shot.taken_at, shot.id));
    let mut groups = Vec::new();
    let mut current = Vec::new();
    let mut previous: Option<&Shot> = None;
    for shot in ordered {
        let joins = previous.is_some_and(|before| distance(shot.look, before.look) <= max_distance);
        if !joins && current.len() > 1 {
            groups.push(std::mem::take(&mut current));
        }
        if !joins {
            current.clear();
        }
        current.push(shot.id);
        previous = Some(shot);
    }
    if current.len() > 1 {
        groups.push(current);
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    fn picture(shift: u32, flip: bool) -> RgbaImage {
        RgbaImage::from_fn(90, 80, |x, y| {
            let level = ((x + shift) * 255 / 100) as u8 ^ if y > 40 { 0x40 } else { 0 };
            let level = if flip { 255 - level } else { level };
            Rgba([level, level, level, 255])
        })
    }

    #[test]
    fn photos_alike_have_close_looks_and_different_ones_far() {
        let base = look(&picture(0, false));
        assert!(distance(base, look(&picture(2, false))) <= DEFAULT_DISTANCE, "a slightly moved frame");
        assert!(distance(base, look(&picture(0, true))) > DEFAULT_DISTANCE, "a different picture");
        assert_eq!(distance(5, 5), 0);
        assert_eq!(distance(0, u64::MAX), 64);
    }

    #[test]
    fn a_burst_is_close_in_time_and_alike() {
        let shot = |id, taken_at, look| Shot { id, taken_at, look };
        let shots = [
            shot(1, 100, 0b1111),
            shot(2, 101, 0b1110),
            shot(3, 103, 0b1100),
            // Alike but a minute later: a new start.
            shot(4, 170, 0b1100),
            // Right after, but nothing like it.
            shot(5, 171, u64::MAX),
            shot(6, 172, u64::MAX - 1),
            // No time: never grouped.
            shot(7, 0, 0b1111),
        ];
        assert_eq!(group(&shots, 10, DEFAULT_DISTANCE), vec![vec![1, 2, 3], vec![5, 6]]);
        assert_eq!(group(&shots, 120, DEFAULT_DISTANCE), vec![vec![1, 2, 3, 4], vec![5, 6]], "a wider window");
        assert_eq!(group(&shots, DEFAULT_WINDOW, 0), Vec::<Vec<i64>>::new(), "only identical looks");
    }

    #[test]
    fn stack_modes_group_time_or_appearance() {
        let shots = [
            Shot { id: 1, taken_at: 100, look: 0 },
            Shot { id: 2, taken_at: 200, look: 1 },
            Shot { id: 3, taken_at: 1_000, look: 3 },
        ];
        assert_eq!(group_by_time(&shots, 300), vec![vec![1, 2]]);
        assert_eq!(group_by_look(&shots, 1), vec![vec![1, 2, 3]]);
    }

    #[test]
    fn default_time_window_is_five_minutes_inclusive() {
        let shots = [
            Shot { id: 1, taken_at: 1_000, look: 0 },
            Shot { id: 2, taken_at: 1_000 + DEFAULT_WINDOW, look: 0 },
            Shot { id: 3, taken_at: 1_000 + 2 * DEFAULT_WINDOW + 1, look: 0 },
        ];
        assert_eq!(DEFAULT_WINDOW, 300);
        assert_eq!(group_by_time(&shots, DEFAULT_WINDOW), vec![vec![1, 2]]);
    }
}

