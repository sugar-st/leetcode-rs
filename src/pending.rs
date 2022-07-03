#![allow(dead_code)]
#![allow(unused_variables)]

// 1030: https://leetcode.cn/problems/matrix-cells-in-distance-order/
pub fn all_cells_dist_order(rows: i32, cols: i32, r_center: i32, c_center: i32) -> Vec<Vec<i32>> {
    panic!("");
}
// 1037: https://leetcode.cn/problems/valid-boomerang/
pub fn is_boomerang(p: Vec<Vec<i32>>) -> bool {
    // math: vector(ab) x vector(bc) != 0
    (p[0][0] - p[1][0]) * (p[0][1] - p[2][1]) != (p[0][0] - p[2][0]) * (p[0][1] - p[1][1])
}
// 1046: https://leetcode.cn/problems/last-stone-weight/
pub fn last_stone_weight(stones: Vec<i32>) -> i32 {
    panic!("")
}
// 1051: https://leetcode.cn/problems/height-checker/
pub fn height_checker(heights: Vec<i32>) -> i32 {
    // sort and compare
    let mut sorted = heights.clone();
    sorted.sort();
    sorted.into_iter().zip(heights.into_iter()).fold(
        0,
        |res, (new, old)| {
            if new != old {
                res + 1
            } else {
                res
            }
        },
    )
}
// 1089: https://leetcode.cn/problems/duplicate-zeros/
pub fn duplicate_zeros(nums: &mut Vec<i32>) {
    // simulation started from behind
    let mut cnt = nums
        .iter()
        .fold(0, |cnt, x| if *x == 0 { cnt + 1 } else { cnt });
    for i in (0..nums.len()).rev() {
        if nums[i] == 0 {
            cnt -= 1;
        } else if i + cnt <= nums.len() {
            nums[i + cnt] = nums[i];
        }
    }
}
// 1122: https://leetcode.cn/problems/relative-sort-array/
pub fn relative_sort_array(arr1: Vec<i32>, arr2: Vec<i32>) -> Vec<i32> {
    use std::collections::HashMap;
    let m = arr2
        .into_iter()
        .enumerate()
        .fold(HashMap::new(), |mut m, (idx, arr)| {
            m.insert(arr, idx);
            m
        });
    let mut arr1 = arr1;
    arr1.sort_by(|a, b| {
        use std::cmp::Ordering;
        match (m.get(a), m.get(b)) {
            (Some(i), Some(j)) => i.partial_cmp(j).unwrap(),
            (Some(_), None) => Ordering::Greater,
            (None, Some(_)) => Ordering::Less,
            (None, None) => a.partial_cmp(b).unwrap(),
        }
    });
    arr1
}
// 1128: https://leetcode.cn/problems/number-of-equivalent-domino-pairs/
pub fn num_equiv_domino_pairs(dominoes: Vec<Vec<i32>>) -> i32 {
    panic!("");
}
// 1160: https://leetcode.cn/problems/find-words-that-can-be-formed-by-characters/
pub fn count_characters(words: Vec<String>, chars: String) -> i32 {
    fn frequency(word: String) -> [i8; 26] {
        word.as_bytes().into_iter().fold([0; 26], |mut freq, &b| {
            freq[b as usize - 'a' as usize] += 1;
            freq
        })
    }
    let chars = frequency(chars);
    words.into_iter().fold(0, |res, word| {
        let cnt = word.len() as i32;
        let freq = frequency(word);
        for i in 0..26 {
            if freq[i] > chars[i] {
                return res;
            }
        }
        res + cnt
    })
}
